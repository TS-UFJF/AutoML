import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.model_selection import train_test_split
import pandas as pd
from .BaseWrapper import BaseWrapper


class TFTWrapper(BaseWrapper):
    def __init__(self, quantiles):
        super().__init__(quantiles)
        self.intern_training = None
        self._intern_validation = None
        self.training = None
        self.validation = None
        self.model = None
        self.trainer = None
        self.oldest_lag = None
        self.last_period = None
        self.quantiles = quantiles

    def transform_data(self, data, past_lags, index_label, target_label, train_val_split):

        self.past_lags = past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = index_label
        self.target_label = target_label

        # External train and validation sets
        X = data[[index_label]]
        y = data[[target_label]]

        self.training = (X.loc[:int(len(data) * train_val_split)], y.loc[:int(len(data) * train_val_split)])
        self.validation =(X.loc[int(len(data) * train_val_split):], y.loc[int(len(data) * train_val_split):])

        # intern train and validation sets, they use dataloaders to optimize the training routine
        # time index are epoch values
        # data["time_idx"] = (data[self.index_label] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        data["time_idx"] = data.index
        data['group_id'] = 'series'

        max_prediction_length = self.oldest_lag
        max_encoder_length = self.oldest_lag
        # training_cutoff = data["time_idx"].max() - max_prediction_length

        self.intern_training = TimeSeriesDataSet(
            data[:int(len(data) * train_val_split)],
            time_idx="time_idx",
            group_ids=["group_id"],
            target=self.target_label,
            min_encoder_length=0,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group_id"],
            # time_varying_unknown_reals=[self.target_label],
            # the docs says that the max_lag < max_encoder_length
            # lags={self.target_label: list(self.past_lags[1:-1] + 1)},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            # allow_missings=True
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self._intern_validation = TimeSeriesDataSet.from_dataset(
            self.intern_training, data, predict=True, stop_randomization=True)

        # store the last input to use as encoder data to next predictions
        self.last_period = data.iloc[-(self.oldest_lag*2+1):].copy()

    def train(self,
              max_epochs=25,
              hidden_size=16,
              lstm_layers=1,
              dropout=0.1,
              attention_head_size=4,
              reduce_on_plateau_patience=4,
              hidden_continuous_size=8,
              learning_rate=1e-3,
              gradient_clip_val=0.1,
              ):
        # configure network and trainer
        # create dataloaders for model
        batch_size = 128
        train_dataloader = self.intern_training.to_dataloader(
            train=True, batch_size=batch_size)
        val_dataloader = self._intern_validation.to_dataloader(
            train=False, batch_size=batch_size * 10)

        pl.seed_everything(42)

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        # lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            weights_summary=None,
            gradient_clip_val=gradient_clip_val,
            # limit_train_batches=30,  # coment in for training, running validation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[early_stop_callback],
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.intern_training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            lstm_layers=lstm_layers,
            output_size=len(self.quantiles),  # 3 quantiles by default
            loss=QuantileLoss(self.quantiles),
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )

        # res = trainer.tuner.lr_find(
        #     self.model,
        #     train_dataloader=train_dataloader,
        #     val_dataloaders=val_dataloader,
        #     max_lr=10.0,
        #     min_lr=1e-6,
        # )

        # self.model = TemporalFusionTransformer.from_dataset(
        #     self.intern_training,
        #     learning_rate=res.suggestion(), # using the suggested learining rate
        #     hidden_size=hidden_size,
        #     attention_head_size=attention_head_size,
        #     dropout=dropout,
        #     hidden_continuous_size=hidden_continuous_size,
        #     output_size=len(self.quantiles),  # 3 quantiles by default
        #     loss=QuantileLoss(self.quantiles),
        #     reduce_on_plateau_patience=reduce_on_plateau_patience,
        # )

        # fit network
        trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def _auto_feed(self, X, future_steps, quantile=False):
        """
        Perform autofeed over the X values to predict the futures steps.
        """

        def append_new_data(cur_X, new_value, date_step):
            new_date = cur_X[self.index_label].iloc[-1] + date_step
            new_entry = {
                self.index_label: new_date,
                self.target_label: new_value,
                'time_idx': cur_X['time_idx'].iloc[-1] + 1,
                'group_id': 'series'
            }
            return cur_X.append(new_entry, ignore_index=True)


        # prediction or quantile mode
        mode = 'quantiles' if quantile else 'prediction'

        # interval between dates (last two dates in the dataset)
        cur_X = X.copy()
        date_step = cur_X[self.index_label].iloc[-1] - \
            cur_X[self.index_label].iloc[-2]

        y = []

        # if the future steps is less or equals than the oldest lag the model can predict it by default
        if future_steps <= self.oldest_lag:
            predict = self.model.predict(cur_X, mode=mode)[0].numpy().tolist()
            return predict[:future_steps]
        else:
            # short cut the auto feed prediction with more reliable prediction
            predict = self.model.predict(cur_X, mode=mode)[0].numpy().tolist()
            for new_value in predict:
                cur_X = append_new_data(cur_X, new_value, date_step)
            y = predict


        for _ in range(self.oldest_lag, future_steps):
            predict = self.model.predict(cur_X, mode=mode)[0][0]
            if quantile:
                y.append(predict.numpy().tolist())
                new_value = y[-1][1]  # get quantil 0.5
            else:
                y.append(float(predict.numpy()))
                new_value = y[-1]

            cur_X = append_new_data(cur_X, new_value, date_step)

        return y

    def _verify_target_column(self, data):
        if not self.target_label in data.columns:
            data[self.target_label] = 0

    def predict(self, X, future_steps, history, quantile=False):
        predictions = []

        self._verify_target_column(X)

        for i in range(len(X)):
            X_temp = history.append(X.iloc[:i], ignore_index=True)
            time_idx = list(range(len(X_temp)))  # refact to use real time idx
            time_idx = [idx + self.last_period["time_idx"].max()
                        for idx in time_idx]
            X_temp[self.index_label] = pd.to_datetime(X_temp[self.index_label])
            X_temp[self.index_label] = X_temp[self.index_label].dt.tz_localize(None)
            X_temp["time_idx"] = time_idx
            X_temp['group_id'] = 'series'

            y = self._auto_feed(X_temp, future_steps, quantile)
            predictions.append(y)

        return predictions

    def next(self, X, future_steps, quantile=False):

        self._verify_target_column(X)

        # pre-process the data
        X[self.index_label] = pd.to_datetime(X[self.index_label])
        X[self.index_label] = X[self.index_label].dt.tz_localize(None)
        X['group_id'] = 'series'

        temp_data = self.last_period.iloc[-(self.oldest_lag+1):].copy()

        cur_X = temp_data.append(X, ignore_index=True)
        time_idx = list(range(len(cur_X)))  # refact to use real time idx
        cur_X["time_idx"] = time_idx

        cur_X.index = list(range(len(cur_X)))

        y = self._auto_feed(cur_X, future_steps, quantile)

        return y

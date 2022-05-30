from tools.faceAttr_trainer import Classifier_Trainer
from tools import config as cfg

trainer = Classifier_Trainer(cfg.epochs, cfg.batch_size, cfg.lr, cfg.model_type)
trainer.fit()

import pandas as pd 
import os
import numpy as np



@hydra.main(version_base=None, config_path='../configs', config_name="config")
def ihs_message_dim(cfg : DictConfig) -> None:
    
    notification_path = cfg.ihs.notifications
    messagedim_path = cfg.ihs.save_path
    # "data/IHS/raw/2023/intervention/notification_sent_2023.csv"
    notifications = pd.read_csv(notification_path)
    #"data/IHS/processed/2023/message_dimensions.csv"
    message_dims =  pd.read_csv(messagedim_path)

    notifications.columns  = notifications.columns.str.lower()
    message_dims.columns  = message_dims.columns.str.lower().str.replace(" ", "")

    models = os.listdir(cfg.ihs.model_prefix)


    message_dims = message_dims[['notificationidentifier'] + models]
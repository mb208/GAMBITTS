# Multiple Imputation for IHS  data 
  # 2024-12-27 - Imputating mood will move on to steps next 
  # 2025-01-20 - Imputed for mood, sleep, and steps # reduces number of pctps to 611
# 
# Authored by Marc Brooks


library(mice)
library(tidyverse)
library(jsonlite)

## setting
imputation_method = "pmm"
min_date = as.Date("2022-07-01")
max_date = as.Date("2023-06-30")
random_seed = 20210902
set.seed(random_seed)
min_sleep = 30
max_sleep = 1440
# M = 20 # number of iterations for imputation (Doesn't matter for NATS)


srvy_data = read_csv("./data/IHS/raw/2022/survey/IHSdata_2022_08112023.csv")
mood_data = read_csv("./data/IHS/raw/2022/mood/mood_daily_2022.csv")
steps_data = read_csv("./data/IHS/raw/2022/wearable/step/daily_steps_both.csv")
# fitbit sleep
sleep_dataFB =  read_csv("./data/IHS/raw/2022/wearable/sleep/fitbit_daily_sleep_2022.csv") %>% 
  select(UserID, date, sleep)
#health kit sleep
sleep_dataHK =  read_csv("./data/IHS/raw/2022/wearable/sleep/healthkit_daily_sleep_2022.csv") %>% 
  select(UserID, date, sleep = asleep)

sleep_data <- rbind(sleep_dataFB, sleep_dataHK)

intervention_ids =  read_csv("./data/IHS/raw/2022/intervention/InterventionMessages_sent_2022.csv") %>%
  rename(notificationidentifier=identifier)


# msg_data = read_csv("./data/raw/2022/intervention/2022 IHS Messages FOR ANALYSIS.csv") %>% 
#   select(identifier = `Notification Identifier`, core_msg_id = `Core Message ID`,
#          data_cat = `Data Category`, msg_type = `Message Type`, msg_vrs = `Message Version`) %>% 
#   # This has the same identifier will need to figure out what to do about it later
#   filter(!(core_msg_id==52 & msg_vrs == "Loss Frame")) 


### Util functions
get_curr_mat <- function(tmp, curr_date) {
  tmp %>% filter(date == curr_date) %>% 
    select(-c(date)) %>% 
    rename_with(
    .fn = ~ paste(.x, format(curr_date, "%m%d"), sep = "_"),
    .cols = c(steps_sqrt, sleep, mood)
  )
}


# Process mood data
names(mood_data) <- tolower(names(mood_data))

mood_bl <-  mood_data %>% 
  mutate(
    mood = na_if(mood, 0)) %>% 
  filter(date < min_date) %>% 
  group_by(userid) %>% 
  summarise(pre_intern_mood = mean(mood, na.rm=TRUE))

mood_study <- mood_data %>% 
  mutate(
  mood = na_if(mood, 0)) %>% 
  filter((date >= min_date) & (date <= max_date))

mood_study <- 
  expand.grid(
    userid = unique(mood_data$userid), 
    date = seq(min_date, max_date,1)
  ) %>% 
  left_join(mood_study) %>% 
  tibble()

# Process steps data
names(steps_data) <- tolower(names(steps_data))

steps_bl <- steps_data %>% 
  mutate(
    steps = na_if(steps, 0),
    steps_sqrt = sqrt(steps)) %>% # mtuate to sqrt(steps )
  filter(date < min_date) %>% 
  group_by(userid) %>% 
  summarise(pre_intern_step_sqrt = mean(steps_sqrt, na.rm=TRUE))

steps_study <- steps_data %>% 
  mutate(
    steps = na_if(steps, 0),
    steps_sqrt = sqrt(steps)) %>%  # mtuate to sqrt(steps )
  filter((date >= min_date) & (date <= max_date)) %>% 
  # Some users have recordings from multiple so we'll just randomly take a device 
  group_by(userid, date) %>% 
  summarise(steps_sqrt = last(steps_sqrt), 
            device = last(device))  %>% 
  ungroup()

steps_study <- 
  expand.grid(
    userid = unique(steps_study$userid),  
    date = seq(min_date, max_date,1)
  ) %>%
  left_join(steps_study) %>%
  # mutate(week = as.integer(difftime(date, min(date), units = "weeks"))) %>%
  tibble() 
  
# Process sleep data
names(sleep_data) <- tolower(names(sleep_data))

sleep_bl <- sleep_data %>% 
  mutate(
    sleep = na_if(sleep, 0)) %>% 
  # Some users have multiple recordings of sleep, most seem like they could be 
  # for the same night and add up to  reasonable hour, there are a couple they get to 
  # very high numbers and 1 over 24hrs which we trat as missing 
  group_by(userid, date) %>% 
  summarise(sleep = sum(sleep))  %>% 
  filter(date < min_date) %>% 
  ungroup() %>% 
  mutate(sleep = if_else(((sleep > 1440) | (sleep < 30)), NA_real_ , sleep)) %>%
  group_by(userid) %>% 
  summarise(pre_intern_step = mean(sleep, na.rm=TRUE))

sleep_study <- sleep_data %>% 
  mutate(
    steps = na_if(sleep, 0)) %>% 
  filter((date >= min_date) & (date <= max_date)) %>% 
  # Some users have multiple recordings of sleep, most seem like they could be 
  # for the same night and add up to  reasonable hour, there are a couple they get to 
  # very high numbers and 1 over 24hrs which we trat as missing 
  group_by(userid, date) %>% 
  summarise(sleep = sum(sleep))  %>% 
  ungroup() %>% 
  mutate(sleep = if_else(sleep > 1440, NA , sleep))

sleep_study <- 
  expand.grid(
    userid = unique(sleep_study$userid), 
    date = seq(min_date, max_date,1)
  ) %>%
  left_join(sleep_study) %>%
  mutate(week = as.integer(difftime(date, min(date), units = "weeks"))) %>%
  tibble() 



# Relevant survey data variables
# 'Sex', 'PHQtot0', 'Neu0', 'depr0', 'EFE0’
# Gender, PHQtot0, Neu0, depr0, EFE0

names(srvy_data) <- tolower(names(srvy_data))
srvy_data <- srvy_data %>%
  mutate(
    sex = recode(sex, `1` = "Male", `2` = "Female")
  ) %>%
  select(userid, participantidentifier, sex, phqtot0, neu0, depr0, efe0) %>%
  filter(!is.na(sex))

# Intervention data
names(intervention_ids) <- tolower(names(intervention_ids))
intervention_ids <- intervention_ids %>%
  filter((date >= min_date) & (date <= max_date))  %>% 
  filter(!str_detect(notificationidentifier, "^Pre"))

  
study_intervention_ids <- expand.grid(
  userid = unique(intervention_ids$userid), 
  date = seq(min_date, max_date,1)
) %>%
  left_join(intervention_ids)  %>%
tibble()

# users actually in the study across measurements
userids <- Reduce(intersect, list(unique(srvy_data$userid),
                          unique(mood_data$userid),
                          unique(steps_data$userid),
                          unique(sleep_study$userid),
                          unique(study_intervention_ids$userid)))


steps_study <- steps_study %>% filter(userid %in% userids)
mood_study <- mood_study %>% filter(userid %in% userids)
sleep_study <- sleep_study %>% filter(userid %in% userids)
study_intervention_ids <- study_intervention_ids %>% filter(userid %in% userids)
bl_data <- srvy_data %>% filter(userid %in% userids) %>% 
  left_join(mood_bl) %>% 
  left_join(steps_bl) %>%
  left_join(sleep_bl) %>% 
  select(-participantidentifier)

ihs_data <- steps_study %>% 
  inner_join(mood_study) %>%
  inner_join(sleep_study) %>%
  mutate(year_week = strftime(date, "%Y-%V"),
         week = as.numeric(factor(year_week, levels = unique(year_week)))-1) %>%
  select(-year_week)


max_date <- as.Date(max(ihs_data$date))
impute_list = list()

# for (m in 1:M) {
# print(paste("Starting iteration ", m))
ihs_complete = tibble()
## variable in personal baseline
## "STUDY_PRTCPT_ID","PreInternStep","PreInternSleep","PreInternMood","Gender","PHQtot0","Neu0","depr0","EFE0"
data_impute <- bl_data %>% select(-userid)
mice_output <- mice(data_impute, m = 1, method = imputation_method)
bldat_complete <-
  bl_data %>% select(userid) %>% 
  bind_cols(complete(mice_output, 1))

# 07-01 to 07-03 (First three days )
tmp <- ihs_data %>% 
  right_join(bldat_complete %>% distinct(userid)) %>% 
  arrange(userid,date) %>% select(-c(device, year))

# 07-01
curr_date <- as.Date("2022-07-01")
history_mat <- bldat_complete

curr_mat <- get_curr_mat(tmp, curr_date)

impute_mat <- curr_mat %>%
  left_join(history_mat, by="userid") %>% 
  select(-c(userid, week))

mice_output <- mice(impute_mat, m=1, method = imputation_method)
complete_mat <- curr_mat %>% select(userid, week) %>%
  bind_cols(complete(mice_output, 1))
ihs_complete <- complete_mat %>% 
  select(userid, week, sex, ends_with(format(curr_date, "%m%d"))) %>%
  rename_with(.fn = ~ str_remove(., paste(
    "_", format(curr_date, "%m%d"), sep = ""
  ))) %>%
  add_column(date = curr_date, .after = "userid") %>%
  bind_rows(ihs_complete)

history_mat <- history_mat %>% select(userid) %>% 
  bind_cols(complete(mice_output, 1))

# 07-02
curr_date <- as.Date("2022-07-02")
curr_mat <- get_curr_mat(tmp, curr_date)

impute_mat <- curr_mat %>%
  left_join(history_mat, by="userid") %>% 
  select(-c(userid, week))

mice_output <- mice(impute_mat, m=1, method = imputation_method)
complete_mat <- curr_mat %>% select(userid, week) %>%
  bind_cols(complete(mice_output, 1))
ihs_complete <- complete_mat %>% 
  select(userid,  week, sex, ends_with(format(curr_date, "%m%d"))) %>%
  rename_with(.fn = ~ str_remove(., paste(
    "_", format(curr_date, "%m%d"), sep = ""
  ))) %>%
  add_column(date = curr_date, .after = "userid") %>%
  bind_rows(ihs_complete)

history_mat <- history_mat %>% select(userid) %>% 
  bind_cols(complete(mice_output, 1))

# 07-03
curr_date <- as.Date("2022-07-03")
curr_mat <- get_curr_mat(tmp, curr_date)

impute_mat <- curr_mat %>%
  left_join(history_mat, by="userid") %>% 
  select(-c(userid, week))

mice_output <- mice(impute_mat, m=1, method = imputation_method)
complete_mat <- curr_mat %>% select(userid, week) %>%
  bind_cols(complete(mice_output, 1))
ihs_complete <- complete_mat %>% 
  select(userid, week, sex, ends_with(format(curr_date, "%m%d"))) %>%
  rename_with(.fn = ~ str_remove(., paste(
    "_", format(curr_date, "%m%d"), sep = ""
  ))) %>%
  add_column(date = curr_date, .after = "userid") %>%
  bind_rows(ihs_complete)
history_mat <- history_mat %>% select(userid) %>% 
  bind_cols(complete(mice_output, 1))
  
  
# Running imputation for rest of the trial
study_dates = seq(as.Date("2022-07-04"), max_date, 1)
for (curr_date in as.list(study_dates)) {
  ## need to finish for loop for rest of imputation
  curr_mat <- get_curr_mat(tmp, curr_date)
  curr_week = curr_mat$week[1]
  if ((weekdays(curr_date) == "Monday")) {
    tmp_data = ihs_complete %>% filter(week == curr_week - 1) %>%
      group_by(userid) %>% summarise(
        prev_week_step = mean(steps_sqrt),
        prev_week_sleep = mean(sleep),
        prev_week_mood = mean(mood)
      )
    history_mat <-
      history_mat %>% select(-starts_with("prev_week")) %>%
      left_join(tmp_data)
  }
  
  impute_mat <- curr_mat %>%
    left_join(history_mat, by="userid") %>% 
    select(-c(userid, week))
  mice_output <- mice(impute_mat, m=1, method = imputation_method)
  complete_mat <- curr_mat %>% select(userid, week) %>%
    bind_cols(complete(mice_output, 1))
  
  # browser()
  ihs_complete <- complete_mat %>% 
    select(userid,  week, sex, ends_with(format(curr_date, "%m%d"))) %>%
    rename_with(.fn = ~ str_remove(., paste(
      "_", format(curr_date, "%m%d"), sep = ""
    ))) %>%
    add_column(date = curr_date, .after = "userid") %>%
    bind_rows(ihs_complete)
  
  history_mat <- complete_mat %>% select(-ends_with(format(curr_date - 3, "%m%d"))) %>% 
    select(-week)
}
  

#   impute_list[[m]] <- list(
#     "complete_data"= ihs_complete,
#     "original_data" = ihs_data,
#     "baseline_original_data" = bl_data,
#     "baseline_complete_data" = bldat_complete
#   )
#   print(paste("Ending iteration ", m))
# }
# impute_list["seed"] = random_seed
# impute_list["num_imputations"] = M

# Rdata file 
# Naming to indicate users with many NAs were filtered.
# save(impute_list, file = "./data/processed/2022_23_ihs_imputed_wout_largeNA.Rdata")

intervention_ids$date = as.Date(intervention_ids$date)
ihs_complete <- ihs_complete %>% inner_join(intervention_ids, by=c("userid", "date"))
write_csv(ihs_complete,  file = "./data/IHS/imputed/2022/2022_23_ihs_imputed.csv")



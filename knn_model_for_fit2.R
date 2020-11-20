library(janitor)
library(tidyverse)
library(tidymodels)
library(rsample)
library(recipes)
library(parsnip)
library(kknn)

# workflow
set.seed(3000)
full_train <- read_csv("data/train.csv") %>% 
  mutate(classification = factor(classification,
                                 levels = 1:4,
                                 labels = c("far below", "below", "meets", "exceeds"),
                                 ordered = TRUE))

data <- dplyr::sample_frac(full_train, size = 0.005)

# take a look at the outcome
data %>% tabyl(classification)

# create initial splits 
set.seed(3000)
data_split <- initial_split(data)

set.seed(3000)
train_split <- training(data_split)
test_split <- testing(data_split)

# use the training data to create a k-fold cross-validation data object
set.seed(3000)
cv_splits <- vfold_cv(train_split)

# create recipe
rec <- 
  recipe(classification ~ enrl_grd + lat + lon + gndr, data = train_split) %>%
  step_mutate(enrl_grd = factor(enrl_grd), gndr = factor(gndr)) %>%
  step_meanimpute(lat, lon) %>%
  step_unknown(enrl_grd, gndr) %>% 
  step_dummy(enrl_grd, gndr) %>%  
  step_normalize(lat, lon)

# set a tuned KNN model
knn_mod <- 
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           weight_func = tune(),
           dist_power = tune())

# parallel processing
parallel::detectCores()

cl <- parallel::makeCluster(8)
doParallel::registerDoParallel(cl)

knn_res <- tune::tune_grid(
  knn_mod,
  preprocessor = rec,
  resamples = cv_splits,
  control = tune::control_resamples(save_pred = TRUE) #here we save predictions
)

parallel::stopCluster(cl)

# take a look at our predictions amd metrics
collect_predictions(knn_res)
collect_metrics(knn_res)

# take a look at the 3 parameters we tuned
knn_res %>% 
  collect_metrics(summarize = FALSE) %>% 
  distinct(neighbors, weight_func, dist_power)

# how about the model performance? take a look at auc and accuracy
knn_res %>% 
  show_best(metric = "roc_auc", n = 5)
# poor performance.... mean is around .5, meaning as good as by chance
knn_res %>% 
  show_best(metric = "accuracy", n = 5)
# poor performance.... mean is around .3, meaning worse than by chance

knn_res %>% 
  autoplot()+
  geom_line()

# we will use this model anyways
knn_best <- select_best(knn_res, "roc_auc")
  
final_mod <- knn_mod %>% 
  finalize_model(knn_best)

final_rec <- rec %>% 
  finalize_recipe(knn_best)

final_mod

final_rec

# run the final fit on the inital data split
cl <- parallel::makeCluster(8)
doParallel::registerDoParallel(cl)

final_fit <- last_fit(final_mod, 
                      preprocessor = final_rec, 
                      split = data_split)

parallel::stopCluster(cl)

# collect metrics
final_fit %>% 
  collect_metrics()

collect_metrics(final_fit)

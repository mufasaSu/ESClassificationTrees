# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
setwd("/Users/musto/Documents/Masterarbeit/RandomForest")

# Load and prepare data
data_depth <- read.csv(file.path(getwd(), "experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/scikit_rf_depth_vs_acc.csv"))

# Print structure of data_depth to see available columns
print("Columns in data_depth:")
print(names(data_depth))

# Add diagnostic print after initial filtering
print("Datasets after initial filtering:")
print(unique(data_depth$dataset))

# Drop unnecessary columns
data_depth <- data_depth %>% 
  select(-n_samples, -mc_iterations, -feature_dim)

# Split into train and test data
train_data <- data_depth %>%
  select(dataset, starts_with("Train.Accuracy.for.Depth")) %>%
  rename_with(~gsub("Train Accuracy for Depth ", "Train.Accuracy.Depth.", .))

test_data <- data_depth %>%
  select(dataset, starts_with("Test.Accuracy.for.Depth")) %>%
  rename_with(~gsub("Test Accuracy for Depth ", "Test.Accuracy.Depth.", .))



# Prepare data for plotting
train_long <- train_data %>%
  pivot_longer(cols = -dataset,  # Select all columns except 'dataset'
               names_to = "Depth",
               values_to = "Train_Accuracy") %>%
  mutate(Depth = as.numeric(gsub(".*Depth\\.", "", Depth)))

test_long <- test_data %>%
  pivot_longer(cols = -dataset,  # Select all columns except 'dataset'
               names_to = "Depth",
               values_to = "Test_Accuracy") %>%
  mutate(Depth = as.numeric(gsub(".*Depth\\.", "", Depth)))

# Combine the datasets
combined_data <- merge(train_long, test_long, by = c("dataset", "Depth"))

# Load and prepare ES* data
data_es <- read.csv(file.path(getwd(), "experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_dgps_results_mean_median.csv"), check.names=FALSE)

# Filter to get only rows where dgp_config_folder is "standard" and algorithm_name is " UES*"
data_es <- data_es %>% 
  filter(dgp_config_folder == "standard", algorithm_name == "UES*") %>%
  select(dataset, `median_depth (median)`) %>%
  rename(ES.Single.Depth.Sqrt = `median_depth (median)`)

# Modify the plotting function to handle only ES* median depths
plot_data <- function(data, es_data, nrow = NULL, ncol = NULL, title = "Train vs Test Accuracies") {
  # Calculate default nrow and ncol if not provided
  if (is.null(nrow) || is.null(ncol)) {
    n_datasets <- length(unique(data$dataset))
    ncol <- 4
    nrow <- ceiling(n_datasets / ncol)
  }
  
  # Create color values
  color_values <- c(
    "Train Accuracy" = "royalblue4",
    "Test Accuracy" = "tomato4",
    "URES* Depth" = "olivedrab3"
  )
  
  # Create linetype values
  linetype_values <- c(
    "Train Accuracy" = "solid",
    "Test Accuracy" = "solid",
    "URES* Depth" = "dashed"
  )
  
  # Create guide override sizes
  guide_sizes <- c(1, 1, 0.8)

  ggplot(data, aes(x = Depth)) +
    geom_line(aes(y = Train_Accuracy, color = "Train Accuracy"), size = 1) +
    geom_line(aes(y = Test_Accuracy, color = "Test Accuracy"), size = 1) +
    geom_vline(data = es_data, 
               aes(xintercept = ES.Single.Depth.Sqrt, 
                   color = "URES* Depth"),
               linetype = "dashed", size = 0.8) +
    labs(
      x = "Depth",
      y = "Accuracy",
      color = "Overview"
    ) +
    scale_y_continuous(limits = c(0.5, 1)) +
    scale_x_continuous(limits = c(1, 35), breaks = c(1, 5, 10, 15, 20, 25, 30, 35)) +
    scale_color_manual(values = color_values) +
    scale_linetype_manual(values = linetype_values) +
    guides(
      color = guide_legend(override.aes = list(size = guide_sizes)),
      linetype = guide_legend(override.aes = list(size = guide_sizes))
    ) +
    theme_light() +
    theme(
      text = element_text(family = "serif"),
      # plot.title = element_text(
      #   size = 24,
      #   color = "black",
      #   hjust = 0.5,
      #   margin = margin(b = 10)
      # ),
      # plot.title.position = "panel",  # Ensure the title aligns with the panel
      # Legend details
      legend.position = "right",
      legend.key.height = unit(1.5, "lines"),
      legend.text = element_text(size = 15),
      legend.title = element_text(size = 18, face = "bold", color = "black"),
      strip.background = element_rect(fill = "lightgray", color = "black"),
      strip.text = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x = element_text(size = 18),  # Increase x-tick text size
      axis.text.y = element_text(size = 18),  # Increase y-tick text size
      axis.title.x = element_text(margin = margin(t = 15), size=18),
      axis.title.y = element_text(margin = margin(r = 15), size=18),
      panel.spacing.x = unit(0.2, "cm")  # Increase horizontal spacing between plots
    ) +
    facet_wrap(~ dataset, nrow = nrow, ncol = ncol)
}

# Create and save the plot
plot_2x4 <- plot_data(combined_data, data_es)
ggsave("latex/thesis/images/rf_URES_star_depth_vs_accuracy.png", plot = plot_2x4, width = 15, height = 8, dpi = 600)

# Print plots
print(plot_2x4)  # One plot with 2 rows, 4 columns

alg_prefix <- "rf_star"

# Save the plots with dynamic filenames
ggsave(sprintf("latex/thesis/images/%s%s_4.png", alg_prefix),
       plot = plot_2x4, width = 14.08, height = 8.67, dpi = 300)
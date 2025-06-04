# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# Set flags for filtering and algorithm selection
FILTER_DIMENSION <- 5  # Set to 2, 5, or NULL
SHOW_ALGORITHMS <- c("ES*")  # Set to c("ES*") for only ES*, or c("ES*", "ES") for both
# SHOW_ALGORITHMS <- c("ES*")

# Load and prepare data
# Use relative paths from the project root
data_depth <- read.csv("experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/depth_analysis.csv")

# Print structure of data_depth to see available columns
print("Columns in data_depth:")
print(names(data_depth))

# Filter dimensions if flag is set
if (!is.null(FILTER_DIMENSION)) {
  data_depth <- data_depth %>% 
    filter(feature_dim != FILTER_DIMENSION)  # Remove the dgp_config_folder filter here
}

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
data <- read.csv("experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv")
if (!is.null(FILTER_DIMENSION)) {
  data <- data %>% 
    filter(dgp_config_folder == "standard") %>%  # Keep this filter for ES* data
    filter(feature_dim != FILTER_DIMENSION)
}

# Add diagnostic print for ES* data
print("Datasets in ES* data after filtering:")
print(unique(data$dataset))

# Simplify ES* data to include selected algorithms
data_es <- data %>% 
  filter(algorithm %in% SHOW_ALGORITHMS) %>%
  select(dataset, algorithm, Depth..median.) %>%
  distinct() %>%
  group_by(dataset) %>%
  mutate(
    # Only subtract 0.1 when values are identical, but don't subtract 1
    Depth..median. = case_when(
      n() > 1 & 
      n_distinct(Depth..median.) == 1 & 
      algorithm == "ES" ~ Depth..median. - 0.1,
      TRUE ~ Depth..median.
    )
  ) %>%
  ungroup()

# Add these diagnostic lines before plotting
print("Number of unique datasets in combined_data:")
print(length(unique(combined_data$dataset)))
print("Unique datasets in combined_data:")
print(unique(combined_data$dataset))

print("Number of unique datasets in data_es:")
print(length(unique(data_es$dataset)))
print("Unique datasets in data_es:")
print(unique(data_es$dataset))

# Standardize dataset names
combined_data <- combined_data %>%
  mutate(dataset = ifelse(dataset == "Smooth", "Circular Smooth", dataset))

# Also standardize dataset names in data_es
data_es <- data_es %>%
  mutate(dataset = ifelse(dataset == "Smooth Signal", "Circular Smooth", dataset))

# Add diagnostic prints to verify the fix
print("After standardization - Unique datasets in combined_data:")
print(unique(combined_data$dataset))
print("Unique datasets in data_es:")
print(unique(data_es$dataset))

# Modify the plotting function to handle multiple algorithms
plot_data <- function(data, es_data, nrow = NULL, ncol = NULL, title = "Train vs Test Accuracies") {
  # Calculate default nrow and ncol if not provided
  if (is.null(nrow) || is.null(ncol)) {
    n_datasets <- length(unique(data$dataset))
    ncol <- 4
    nrow <- ceiling(n_datasets / ncol)
  }
  
  # Create color values dynamically
  color_values <- c(
    "Train Accuracy" = "royalblue4",
    "Test Accuracy" = "tomato4"
  )
  color_values <- c(color_values, setNames(
    c("olivedrab3", "royalblue1")[1:length(SHOW_ALGORITHMS)],
    SHOW_ALGORITHMS
  ))
  
  # Create linetype values dynamically
  linetype_values <- c(
    "Train Accuracy" = "solid",
    "Test Accuracy" = "solid"
  )
  linetype_values <- c(linetype_values, setNames(
    rep("dashed", length(SHOW_ALGORITHMS)),
    SHOW_ALGORITHMS
  ))
  
  # Create guide override sizes dynamically
  guide_sizes <- c(1, 1, rep(0.8, length(SHOW_ALGORITHMS)))
  
  # Create legend labels with renamed ES* to ES* Depth
  legend_labels <- c("Train Accuracy", "Test Accuracy")
  for (alg in SHOW_ALGORITHMS) {
    if (alg == "ES*") {
      legend_labels <- c(legend_labels, "ES* Depth")
    } else {
      legend_labels <- c(legend_labels, alg)
    }
  }
  names(legend_labels) <- c("Train Accuracy", "Test Accuracy", SHOW_ALGORITHMS)

  ggplot(data, aes(x = Depth)) +
    geom_line(aes(y = Train_Accuracy, color = "Train Accuracy"), size = 1) +
    geom_line(aes(y = Test_Accuracy, color = "Test Accuracy"), size = 1) +
    geom_vline(data = es_data, 
               aes(xintercept = Depth..median., 
                   color = algorithm),
               linetype = "dashed", size = 0.8) +
    labs(
      x = "Depth",
      y = "Accuracy",
      color = "Overview"
    ) +
    scale_y_continuous(limits = c(0.5, 1)) +
    scale_x_continuous(limits = c(0, 24), breaks = c(0, 5, 10, 15, 20, 24)) +
    scale_color_manual(values = color_values, labels = legend_labels) +
    scale_linetype_manual(values = linetype_values) +
    guides(
      color = guide_legend(override.aes = list(size = guide_sizes)),
      linetype = guide_legend(override.aes = list(size = guide_sizes))
    ) +
    theme_light() +
    theme(
      text = element_text(family = "serif"),
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
      axis.title.y = element_text(margin = margin(r = 15), size=18)
    ) +
    facet_wrap(~ dataset, nrow = nrow, ncol = ncol)
}

# Create and save the plot
plot_2x4 <- plot_data(combined_data, data_es)
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dt_accuracy_depth.png", plot = plot_2x4, width = 15, height = 8, dpi = 300)
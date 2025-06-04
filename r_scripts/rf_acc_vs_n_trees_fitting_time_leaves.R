# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)

data <- read.csv("experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_dgps_results_mean_median.csv", check.names = FALSE)

# Only keep rows where algorithm is "MD_scikit", "UES*", "IES*" or "CCP"
data <- data[!data$algorithm_name %in% c("UES* (time)", "ICCP", "MD_scikit"), ]

# Filter data to match the Python code
data <- data[data$dgp_config_folder == "standard", ]
data <- data[data$mc_iterations == 300, ]
# d = 2 not part of this!


# Rename UES* to URES* to match the Python code
# UES* has to be renamed!
data$algorithm_name <- gsub("UES\\*", "URES\\*", data$algorithm_name)


# Identify all the test accuracy columns
acc_cols <- grep("test_acc_[0-9]+_trees \\(median\\)", names(data), value = TRUE)

# Reshape data from wide to long format
long_data <- data %>%
  select(algorithm_name, dataset, all_of(acc_cols)) %>%
  pivot_longer(
    cols = all_of(acc_cols),
    names_to = "tree_step",
    values_to = "accuracy"
  )

# Extract the tree number from column names
long_data$tree_number <- as.numeric(gsub("test_acc_([0-9]+)_trees \\(median\\)", "\\1", long_data$tree_step))

# Rename "MD_scikit" to "MD" in the algorithm column
long_data$algorithm_name <- gsub("MD_custom", "MD", long_data$algorithm_name)

# Create the plot
ggplot(long_data, aes(x = tree_number, y = accuracy, color = algorithm_name, group = algorithm_name)) +
  geom_line() +
  facet_wrap(~ dataset, ncol = 4, nrow = 2) +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(
    x = "Number of Trees",
    y = "Test Accuracy",
    color = "Method"
  ) +
  theme_bw() +
    theme_light() +
    theme(
      text = element_text(family = "serif"),
      # Legend details
      legend.position = "right",
      legend.margin = margin(l = 20),  # Add left margin to legend
      # legend.box.background = element_rect(color = "black"),  # Add box around legend
      legend.box.margin = margin(6, 6, 6, 6),  # Add margin around the box
      legend.text = element_text(size = 16),  # Increase legend entry text size
      legend.title = element_text(size = 18, face = "bold"),  # Make legend title bold and larger
      strip.background = element_rect(fill = "lightgray", color = "black"),
      strip.text = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x = element_text(size = 18),  # Increase x-tick text size
      axis.text.y = element_text(size = 18),  # Increase y-tick text size
      axis.title.x = element_text(margin = margin(t = 15), size=18),
      axis.title.y = element_text(margin = margin(r = 15), size=18)
    )
# Save the plot
ggsave("experiments/experiment_analysis/rf_es_simulation_study/plots/rf_acc_vs_n_trees.png", plot = last_plot(), width = 16, height = 8, dpi = 600)
##################


# Read the data
data <- read.csv("experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_dgps_results_mean_median.csv", check.names = FALSE)

# Only keep rows where algorithm is "MD_scikit", "UES*", "IES*" or "CCP"
data <- data[!data$algorithm_name %in% c("UES* (time)", "ICCP", "MD_scikit"), ]

# Filter data to match the Python code
data <- data[data$dgp_config_folder == "standard", ]
data <- data[data$mc_iterations == 300, ]

# Rename UES* to URES* to match the Python code
data$algorithm_name <- gsub("UES\\*", "URES\\*", data$algorithm_name)
# Rename "MD_custom" to "MD"
data$algorithm_name <- gsub("MD_custom", "MD", data$algorithm_name)

# Set factor levels to control the order of bars
data$algorithm_name <- factor(data$algorithm_name, 
                             levels = c("MD", "IGES*", "ILES*", "URES*", "UGES*"))

# Reshape the data to prepare for stacked bars
stacked_data <- data %>%
  select(algorithm_name, dataset, `median_test_acc (median)`, `test_acc (median)`) %>%
  mutate(
    # Calculate difference between test_acc and median_test_acc
    difference = pmax(0, `test_acc (median)` - `median_test_acc (median)`)
  ) %>%
  # Convert to long format with base and difference values
  pivot_longer(
    cols = c(`median_test_acc (median)`, difference),
    names_to = "metric",
    values_to = "value"
  )

# Create proper factor levels for metric to ensure correct stacking order
# Reversing the order to make sure 'difference' is on top
stacked_data$metric <- factor(stacked_data$metric, 
                            levels = c("difference", "median_test_acc (median)"))

# Create an interaction term for algorithm_name and metric
stacked_data$fill_group <- interaction(stacked_data$algorithm_name, stacked_data$metric)

# New approach for colors: create a new variable to determine light vs dark
stacked_data$color_type <- ifelse(stacked_data$metric == "difference", "light", "dark")

# Create a custom fill scale
fill_scale <- c()
for (i in unique(stacked_data$fill_group)) {
  algo <- strsplit(as.character(i), "\\.")[[1]][1]
  metric <- strsplit(as.character(i), "\\.")[[1]][2]
  
  if (algo == "MD") {
    if (metric == "difference") {
      fill_scale[i] <- "#FF9999"  # Light red
    } else {
      fill_scale[i] <- "#CC0000"  # Dark red
    }
  } else {
    if (metric == "difference") {
      fill_scale[i] <- "#66C2A5"  # Light blue-green
    } else {
      fill_scale[i] <- "#1B9E77"  # Dark blue-green
    }
  }
}

# Create the plot with stacked bars
stacked_plot <- ggplot(stacked_data, aes(x = algorithm_name, y = value, fill = fill_group)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~ dataset, ncol = 4, nrow = 2) +
  coord_cartesian(ylim = c(0.5, 1)) +
  scale_fill_manual(values = fill_scale,
                   # Use just two breaks for the legend
                   breaks = c("MD.difference", "MD.median_test_acc (median)"),
                   # Simplify labels to just indicate light/dark without showing colors
                   labels = c("Light - Ensemble", "Dark - Median")) +
  # Use guides() to remove the color rectangles from the legend
  guides(fill = guide_legend(override.aes = list(fill = NA, color = NA))) +
  labs(
    x = "Method",
    y = "Test Accuracy",
    fill = "Type"
  ) +
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    legend.position = "right",
    # Remove the left margin to bring legend closer to plot
    legend.margin = margin(0, 0, 0, 0),  
    # Reduce or remove box margin
    legend.box.margin = margin(0, 0, 0, 0),  
    legend.box.spacing = unit(0, "pt"),  # Remove spacing between legend and plot
    legend.text = element_text(size = 16),  # Increase legend entry text size
    legend.title = element_text(size = 16, face = "bold", hjust = 0.5),  # Center title with hjust = 0.5
    # Remove the legend key background
    legend.key = element_blank(),
    strip.background = element_rect(fill = "lightgray", color = "black"),
    strip.text = element_text(size = 18, face = "bold", color = "black"),
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),  # Decreased size for x-tick labels
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(margin = margin(t = 15), size=18),
    axis.title.y = element_text(margin = margin(r = 15), size=18)
  )

# Print the plot to screen
print(stacked_plot)

# Save the plot
ggsave("experiments/experiment_analysis/rf_es_simulation_study/plots/rf_ensemble_impact.png", plot = last_plot(), width = 14, height = 9, dpi = 300)

# Create a dual-axis plot showing leaves count and fit duration across sample sizes
# Read the data again
data_dual <- read.csv("experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_dgps_results_mean_median.csv", check.names = FALSE)

# Only keep rows where algorithm is "MD_custom" and "UES*"
data_dual <- data_dual[data_dual$algorithm_name %in% c("MD_custom", "UES* (time)"), ]

# Filter for the chosen dataset
data_dual <- data_dual[data_dual$dgp_config_folder == "rf", ]

#rename UES* (Time) to URES*
unique(data_dual$algorithm_name)
data_dual$algorithm_name <- gsub("UES\\* \\(time\\)", "URES\\*", data_dual$algorithm_name)


# chosen_dataset <- "Circular"
# data_dual <- data_dual[data_dual$dataset == chosen_dataset, ]

# Convert n_samples to factor for x-axis and ensure proper ordering
data_dual$n_samples_factor <- factor(data_dual$n_samples,
                                   levels = sort(unique(data_dual$n_samples)))

# Prepare the data for plotting
data_dual$algorithm_name <- factor(data_dual$algorithm_name, levels = c("MD_custom", "URES*"))

# Calculate the overall scaling factor for the dual axis
max_leaves <- max(data_dual$`median_n_leaves (median)`)
max_duration <- max(data_dual$`median_tree_fit_duration (median)`)
scale_factor <- max_leaves / max_duration

# Create the dual-axis plot with sample size as x-axis and algorithm_name as the group/hue
ggplot() +
  # Bars for number of leaves (left y-axis)
  geom_bar(data = data_dual, 
           aes(x = n_samples_factor, y = `median_n_leaves (median)`, fill = algorithm_name),
           stat = "identity", position = position_dodge(width = 0.7), width = 0.6, alpha = 0.8) +
  # Lines connecting fit duration points
  geom_line(data = data_dual,
            aes(x = n_samples_factor, y = `median_tree_fit_duration (median)` * scale_factor,
                color = algorithm_name, group = algorithm_name),
            size = 1.2) +
  # Points for tree fit duration (right y-axis)
  geom_point(data = data_dual,
             aes(x = n_samples_factor, y = `median_tree_fit_duration (median)` * scale_factor,
                 color = algorithm_name), 
             size = 6, shape = 22, stroke = 1.5) +
  # Configure left y-axis (leaves)
  scale_y_continuous(name = "Median Number of Leaves", 
                     sec.axis = sec_axis(~ . / scale_factor * 1000,
                                        name = "Median Fit Duration (ms)")) +
  # Customize colors with separate legends
  scale_fill_manual(name = "Number of Leaves",
                   values = c("MD_custom" = "#1B9E77", "URES*" = "#D95F02"),
                   labels = c("MD_custom" = "MD", "URES*" = "URES*")) +
  scale_color_manual(name = "Median Fit Duration",
                    values = c("MD_custom" = "#1B9E77", "URES*" = "#D95F02"),
                    labels = c("MD_custom" = "MD", "URES*" = "URES*")) +
  # Labels and title
  labs(x = "Sample Size") +
  # Explicitly separate the legends
  guides(fill = guide_legend(order = 1, override.aes = list(alpha = 0.8)),
         color = guide_legend(order = 2, override.aes = list(shape = 22, size = 6, fill = "white"))) +
  # Rest of your theme settings...
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    legend.position = "top",  # Move legend to bottom
    legend.margin = margin(l = 20),  # Add left margin to legend
    legend.box.background = element_rect(color = "black"),  # Add box around legend
    legend.box.margin = margin(6, 6, 6, 6),  # Add margin around the box
    legend.text = element_text(size = 16),  # Increase legend entry text size
    legend.title = element_text(size = 18, face = "bold"),  # Make legend title bold and larger
    axis.text.x = element_text(size = 20),  # Increased from 18
    axis.text.y = element_text(size = 20),  # Increased from 18
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18),
    axis.title.y.right = element_text(margin = margin(l = 15), size = 18)
  )

# Create the dual-axis plot with sample size as x-axis and algorithm_name as the group/hue
ggplot(data_dual, aes(x = n_samples_factor, y = `median_n_leaves (median)`, fill = algorithm_name)) +
  # Bars for number of leaves (left y-axis)
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6, alpha = 0.8) +
  # Lines connecting fit duration points across sample sizes
  geom_line(aes(y = `median_tree_fit_duration (median)` * scale_factor,
                color = algorithm_name, group = algorithm_name),
            size = 1.2, position = position_dodge(width = 0.7)) +
  # Points for tree fit duration (right y-axis)
  geom_point(aes(y = `median_tree_fit_duration (median)` * scale_factor,
                color = algorithm_name, group = algorithm_name), 
             size = 6, shape = 18, stroke = 2, position = position_dodge(width = 0.7)) +
  # Configure left y-axis (leaves)
  scale_y_continuous(name = "Median Number of Leaves", 
                     sec.axis = sec_axis(~ . / scale_factor * 1000,  # Multiply by 1000 to convert to ms
                                        name = "Median Fit Duration (ms)")) +  # Changed unit to ms
  # Customize colors
  scale_fill_manual(values = c("MD_custom" = "#1B9E77", "URES*" = "#D95F02"),
                   labels = c("MD_custom" = "MD", "URES*" = "URES*")) +  # Add labels to rename
  scale_color_manual(values = c("MD_custom" = "#1B9E77", "URES*" = "#D95F02"),
                    labels = c("MD_custom" = "MD", "URES*" = "URES*")) +  # Add labels to rename
  # Labels and title
  labs(
    x = "Sample Size",
    fill = "Method",
    color = "Method"
  ) +
  # Theme consistent with other plots
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    legend.position = "top",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold"),
    legend.margin = margin(t = 10),
    axis.text.x = element_text(size = 20),  # Increased from 18
    axis.text.y = element_text(size = 20),  # Increased from 18
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18),
    axis.title.y.right = element_text(margin = margin(l = 15), size = 18)
  )
# show the plot
print(last_plot())
# save the plot
ggsave("experiments/experiment_analysis/rf_es_simulation_study/plots/rf_leaves_duration_comparison.png", plot = last_plot(), width = 14, height = 10, dpi = 600)

library(dplyr)
library(tidyr)

# Read the CSV file
df <- read.csv("experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv", check.names = FALSE)

# Also standardize dataset names in data_es
df <- df %>%
  mutate(dataset = ifelse(dataset == "Smooth Signal", "Circular Smooth", dataset))

# Apply the filters:
# 1. Keep only rows where dgp_config_folder is "standard"
# 2. Remove rows where feature_dim is 5
# 3. Remove rows where algorithm contains "TS" or "TS*" AND full_alpha_range is "0.0"
# 4. Select only specified columns
filtered_df <- df %>%
  filter(dgp_config_folder == "standard") %>%
  filter(feature_dim != 5) %>%
  filter(!(grepl("TS", algorithm) & full_alpha_range == 1.0)) %>%
  filter(!(grepl("CCP", algorithm) & full_alpha_range == 1.0)) %>%
  filter(algorithm != "MD") %>%
  select(dataset, algorithm, feature_dim, n_train_samples, `Test Accuracy (median)`) %>%
  mutate(`Test Accuracy (median)` = round(`Test Accuracy (median)`, 2))  # Round to 2 decimal places

# Display the result
filtered_df

# Create the plot
library(ggplot2)

# Convert algorithm to factor with specific order
filtered_df$algorithm <- factor(filtered_df$algorithm, 
                              levels = c("CCP", "ES*", "ES", "TS*", "TS"))

ggplot(filtered_df, aes(x = algorithm, y = `Test Accuracy (median)`, fill = algorithm)) +
  geom_bar(stat = "identity") +
  # Add text labels on top of bars, rounded to 2 decimals
  geom_text(aes(label = round(`Test Accuracy (median)`, 2)), 
            vjust = -0.5, size = 4) +
  facet_wrap(~dataset, nrow = 2, ncol = 4) +
  theme_light() +
  labs(y = "Test Accuracy",
       x = "Method") +
  scale_fill_manual(values = c("CCP" = "royalblue4", 
                              "ES*" = "royalblue4", 
                              "ES" = "royalblue4",
                              "TS" = "royalblue4",
                              "TS*" = "royalblue4")) +
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
    legend.position = "none",
    strip.background = element_rect(fill = "lightgray", color = "black"),
    strip.text = element_text(size = 16, face = "bold", color = "black"),
    axis.text.x = element_text(size = 14, angle = 45, hjust = 1),  # Rotated x-axis labels
    axis.text.y = element_text(size = 14),  # Increase y-tick text size
    axis.title.x = element_text(margin = margin(t = 16), size=18),
    axis.title.y = element_text(margin = margin(r = 16), size=18)
    ) +
  coord_cartesian(ylim = c(0.5, 1))

ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dt_test_acc_comp.png", 
       width = 12, 
       height = 7, 
       dpi = 300)


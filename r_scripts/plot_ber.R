library(ggplot2)
library(dplyr)
############################## p = 0.2 ########################################
# Simulate Bernoulli data
set.seed(123)  # For reproducibility
p <- 0.9  # Probability of success
n <- 10000  # Number of trials
bernoulli_data <- data.frame(
  Outcome = rbinom(n, 1, p)
)

# Calculate relative frequency
bernoulli_data <- bernoulli_data %>%
  group_by(Outcome) %>%
  summarize(Frequency = n() / n)

# Create the histogram
histogram <- ggplot(bernoulli_data, aes(x = factor(Outcome), y = Frequency, fill = factor(Outcome))) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    x = "Outcome",
    y = "p",
    fill = ""  # Legend title
  ) +
  ylim(0, 1) +  # Set y-axis limits
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove the legend
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(margin = margin(t = 15), size = 20),
    axis.title.y = element_text(margin = margin(r = 15), size = 20)
  )

# Print the histogram
print(histogram)

ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/ber_09.png", histogram, width = 6.59, height = 4.62, dpi=320)


######## p = 0.8 ########################################
# Simulate Bernoulli data
set.seed(123)  # For reproducibility
p <- 0.4  # Probability of success
n <- 10000  # Number of trials
bernoulli_data <- data.frame(
  Outcome = rbinom(n, 1, p)
)

# Calculate relative frequency
bernoulli_data <- bernoulli_data %>%
  group_by(Outcome) %>%
  summarize(Frequency = n() / n)

# Create the histogram
histogram <- ggplot(bernoulli_data, aes(x = factor(Outcome), y = Frequency, fill = factor(Outcome))) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    x = "Outcome",
    y = "p",
    fill = ""  # Legend title
  ) +
  ylim(0, 1) +  # Set y-axis limits
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove the legend
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(margin = margin(t = 15), size = 20),
    axis.title.y = element_text(margin = margin(r = 15), size = 20)
  )

# Print the histogram
print(histogram)

getwd()
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/ber_04.png", histogram, width = 6.59, height = 4.62, dpi=320)

######## variance curve ########################################
# Create data for the variance curve
x <- seq(0, 1, length.out = 1000)
variance_data <- data.frame(
  p = x,
  variance = x * (1 - x)
)

# Calculate the y-values for the vertical lines
y_04 <- 0.4 * (1 - 0.4)  # y value at x = 0.4
y_09 <- 0.9 * (1 - 0.9)  # y value at x = 0.9

# Create the variance plot
variance_plot <- ggplot(variance_data, aes(x = p, y = variance)) +
  geom_line(size = 1, color = "black") +
  geom_segment(aes(x = 0.4, xend = 0.4, y = 0, yend = y_04), 
              linetype = "dashed", color = scales::hue_pal()(2)[2]) +
  geom_segment(aes(x = 0.9, xend = 0.9, y = 0, yend = y_09), 
              linetype = "dashed", color = scales::hue_pal()(2)[2]) +
  theme_minimal() +
  labs(
    x = "p",
    y = "p(1-p)"
  ) +
  ylim(0, 0.3) +  # Set y-axis limits to show full curve
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(margin = margin(t = 15), size = 16),
    axis.title.y = element_text(margin = margin(r = 15), size = 16)
  )

# Print the plot
print(variance_plot)

ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/ber_var.png", variance_plot, width = 6.59, height = 4.62, dpi=320)

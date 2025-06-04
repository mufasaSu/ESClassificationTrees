# Data Generation Processes (DGPs) for 2D Classification Problems
# This script creates meshgrid plots of the four different DGPs
# to visualize the underlying probability distributions.

library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)
library(grid)

# Function to generate meshgrid for plotting
generate_meshgrid <- function(n_ticks = 100) {
  x1 <- seq(0, 1, length.out = n_ticks)
  x2 <- seq(0, 1, length.out = n_ticks)
  
  # Create all combinations of x1 and x2
  grid <- expand.grid(x1 = x1, x2 = x2)
  return(grid)
}

# 1. Rectangular Classification
rectangular_classification <- function(X, p = 0.8) {
  # X is a data frame with columns x1 and x2
  X_in_rectangular <- (1/3 <= X$x1) & (X$x1 <= 2/3) & 
                      (1/3 <= X$x2) & (X$x2 <= 2/3)
  
  # Calculate probability
  f <- 0.2 + as.integer(X_in_rectangular) * (p - 0.2)
  
  return(f)
}

# 2. Circular Classification
circular_classification <- function(X, p = 0.8) {
  # X is a data frame with columns x1 and x2
  X_in_circular <- sqrt((X$x1 - 1/2)^2 + (X$x2 - 1/2)^2) <= 1/4
  
  # Calculate probability
  f <- 0.2 + as.integer(X_in_circular) * (p - 0.2)
  
  return(f)
}

# 3. Smooth Signal Classification
smooth_signal_classification <- function(X) {
  # X is a data frame with columns x1 and x2
  Z <- exp(-((X$x1 - 0.5)^2 + (X$x2 - 0.5)^2) / 0.1) * 20
  
  # Normalize Z values between 0 and 1
  f <- (Z - min(Z)) / (max(Z) - min(Z))
  
  return(f)
}

# 4. Sine-Cosine Classification
sine_cosine_classification <- function(X) {
  # X is a data frame with columns x1 and x2
  gamma <- 1.5  # Controls the sharpness of the transitions
  f <- 1 / (1 + exp(-gamma * (sin(2 * pi * X$x1) * cos(2 * pi * X$x2))))
  
  return(f)
}

# Create a meshgrid for visualization
grid <- generate_meshgrid(n_ticks = 100)

# Calculate probabilities for each DGP
grid$rectangular <- rectangular_classification(grid)
grid$circular <- circular_classification(grid)
grid$smooth_signal <- smooth_signal_classification(grid)
grid$sine_cosine <- sine_cosine_classification(grid)

# Get default ggplot2 colors in their natural order
default_colors <- scales::hue_pal()(2)
class0_color <- default_colors[1]  # First color in palette (usually reddish)
class1_color <- default_colors[2]  # Second color in palette (usually bluish)

# Create individual plots for each DGP
create_plot <- function(data, value_col, title) {
  ggplot(data, aes_string(x = "x1", y = "x2", fill = value_col)) +
    geom_tile() +
    scale_fill_gradient(
      low = class0_color,
      high = class1_color,
      limits = c(0, 1)
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      breaks = c(0, 0.5, 1),
      labels = c("0", "0.5", "1")
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      expand = c(0, 0),
      breaks = c(0, 0.5, 1),
      labels = c("0", "0.5", "1")
    ) +
    labs(
      title = title,
      x = "X1",
      y = "X2"
    ) +
    coord_fixed(ratio = 1) +
    theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
      axis.text = element_text(size = 18),  # Increased from 14 to 18
      axis.title = element_text(size = 18),  # Increased from 14 to 18
      legend.position = "none",
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "grey95", color = NA),
      strip.background = element_rect(fill = "lightgray", color = "black")
    )
}

# Create plots for each DGP
p1 <- create_plot(grid, "rectangular", "Rectangular")
p2 <- create_plot(grid, "circular", "Circular")
p3 <- create_plot(grid, "smooth_signal", "Circular Smooth")
p4 <- create_plot(grid, "sine_cosine", "Sine Cosine")

# Create a legend
legend_data <- data.frame(x = 1:100/100, y = 1, prob = 1:100/100)
p_legend <- ggplot(legend_data, aes(x = x, y = y, fill = prob)) +
  geom_tile() +
  scale_fill_gradient(
    low = class0_color,
    high = class1_color,
    limits = c(0, 1),
    name = "Probability"
  ) +
  theme(
    legend.title = element_text(size = 22, face = "bold", family = "serif", margin = margin(b = 15)),
    legend.text = element_text(size = 20, family = "serif"),
    legend.key.height = unit(1.5, "cm"),
    legend.key.width = unit(1.5, "cm")
  )

# Extract the legend
legend <- cowplot::get_legend(p_legend)

# Arrange the plots in a grid with shared legend
combined_plot <- gridExtra::grid.arrange(
  p1, p2, p3, p4, legend,
  layout_matrix = rbind(
    c(1, 2, 5),
    c(3, 4, 5)
  ),
  widths = c(0.4, 0.4, 0.2)
)

# Save the combined plot
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/all_dgps_faceted.png", combined_plot, width = 12, height = 10, dpi = 300)

# Also save individual plots for reference
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dgp_rectangular.png", p1, width = 8, height = 6, dpi = 300)
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dgp_circular.png", p2, width = 8, height = 6, dpi = 300)
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dgp_smooth_signal.png", p3, width = 8, height = 6, dpi = 300)
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/dgp_sine_cosine.png", p4, width = 8, height = 6, dpi = 300)

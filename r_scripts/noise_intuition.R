library(ggplot2)
library(grid)      # For adding custom images
library(cowplot)   # For arranging plots
library(png)       # For reading PNG images
#################################################################################
################################## SIMPLE CASE ##################################
#################################################################################

# Define relative width parameter for all plots
# 0.35 gut!
rel_width_ratio <- 0.30  # Adjust this value to change the ratio between colorbar and plot

# Generate two-dimension X which is uniformly distributed in [0, 1] x [0, 1]
X <- data.frame(X1 = runif(1000), X2 = runif(1000))
# Generate y to be 0 for all 1000 rows and set factor levels
X$y <- factor(rep(0, 1000), levels = c(0, 1)) # Ensure levels for both 0 and 1 exist

# Add a dummy row for Class 1 (won't be plotted but ensures legend entry)
X <- rbind(X, data.frame(X1 = NA, X2 = NA, y = factor(1, levels = c(0, 1))))

# Create the scatterplot
scatterplot <- ggplot(X, aes(x = X1, y = X2, color = y)) +
  geom_point(na.rm = TRUE) + # Skip plotting the dummy row
  scale_color_manual(
    values = c(scales::hue_pal()(2)[1], scales::hue_pal()(2)[2]) # Reversed color order
  ) +
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove legend
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18)
  )

# Load the custom colorbar image and wrap it in a ggplot
colorbar_plot <- ggplot() +
  # Set white background first
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 0, 0, 0, unit = "pt")
  ) +
  annotation_custom(
    rasterGrob(
      readPNG("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_scales/no_noise.png", native = TRUE),
      interpolate = TRUE
    )
  )

# First plot_grid (Simple case)
final_plot <- plot_grid(
  colorbar_plot,
  scatterplot,
  ncol = 2,
  rel_widths = c(rel_width_ratio, 1 - rel_width_ratio)
)

# Print the plot
print(final_plot)

# Save the final combined plot
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition_0.png", final_plot, width = 12, height = 8)
getwd()

################################################################################
######################### Bernoullie 1 #######################################
################################################################################

data <- read.csv("data/circular_1.csv")
# Convert y to a factor
data$y <- factor(data$y, levels = c(0, 1)) # Specify levels if needed

# Create the scatterplot
scatterplot <- ggplot(data, aes(x = X1, y = X2, color = y)) +
  geom_point(na.rm = TRUE) + # Skip plotting the dummy row
  scale_color_manual(
    values = c(scales::hue_pal()(2)[1], scales::hue_pal()(2)[2]) # Reversed color order
  ) +
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove legend
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18)
  )

# Load the custom colorbar image and wrap it in a ggplot
colorbar_plot <- ggplot() +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 0, 0, 0, unit = "pt")
  ) +
  annotation_custom(
    rasterGrob(readPNG("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_scales/ber_1.png"), interpolate = TRUE)
  )

# Second plot_grid (Bernoulli 1)
final_plot <- plot_grid(
  colorbar_plot,
  scatterplot,
  ncol = 2,
  rel_widths = c(rel_width_ratio, 1 - rel_width_ratio)
)

# Print the plot
print(final_plot)

ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition_1.png", final_plot, width = 12, height = 8)

################################################################################
######################### Bernoullie 0.8 #######################################
################################################################################

data <- read.csv("data/circular_08.csv")
# Convert y to a factor
data$y <- factor(data$y, levels = c(0, 1)) # Specify levels if needed

# Create the scatterplot
scatterplot <- ggplot(data, aes(x = X1, y = X2, color = y)) +
  geom_point(na.rm = TRUE) + # Skip plotting the dummy row
  scale_color_manual(
    values = c(scales::hue_pal()(2)[1], scales::hue_pal()(2)[2]) # Reversed color order
  ) +
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove legend
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18)
  )

# Load the custom colorbar image and wrap it in a ggplot
colorbar_plot <- ggplot() +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 0, 0, 0, unit = "pt")
  ) +
  annotation_custom(
    rasterGrob(readPNG("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_scales/ber_08.png"), interpolate = TRUE)
  )

# Third plot_grid (Bernoulli 0.8)
final_plot <- plot_grid(
  colorbar_plot,
  scatterplot,
  ncol = 2,
  rel_widths = c(rel_width_ratio, 1 - rel_width_ratio)
)

# Print the plot
print(final_plot)

ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition_2.png", final_plot, width = 12, height = 8)


#################################################################################
################################## CHAOS ########################################
#################################################################################

# Generate two-dimension X which is uniformly distributed in [0, 1] x [0, 1]
X <- data.frame(X1 = runif(1000), X2 = runif(1000))
# Generate y to be bernoullie 0.5 distributed and set factor levels
X$y <- factor(rbinom(n = 1000, size = 1, prob = 0.5), levels = c(0, 1)) # Ensure levels for both 0 and 1 exist

# Add a dummy row for Class 1 (won't be plotted but ensures legend entry)
X <- rbind(X, data.frame(X1 = NA, X2 = NA, y = factor(1, levels = c(0, 1))))

# Create the scatterplot
scatterplot <- ggplot(X, aes(x = X1, y = X2, color = y)) +
  geom_point(na.rm = TRUE) + # Skip plotting the dummy row
  scale_color_manual(
    values = c(scales::hue_pal()(2)[1], scales::hue_pal()(2)[2]) # Reversed color order
  ) +
  theme_light() +
  theme(
    text = element_text(family = "serif"),
    plot.title = element_text(
      size = 22,
      color = "black",
      hjust = 0.5,
      margin = margin(b = 10)
    ),
    legend.position = "none",  # Remove legend
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(margin = margin(t = 15), size = 18),
    axis.title.y = element_text(margin = margin(r = 15), size = 18)
  )

# Load the custom colorbar image and wrap it in a ggplot
colorbar_plot <- ggplot() +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 0, 0, 0, unit = "pt")
  ) +
  annotation_custom(
    rasterGrob(readPNG("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_scales/full_noise.png"), interpolate = TRUE)
  )

# Fourth plot_grid (Chaos)
final_plot <- plot_grid(
  colorbar_plot,
  scatterplot,
  ncol = 2,
  rel_widths = c(rel_width_ratio, 1 - rel_width_ratio)
)

# Print the plot
print(final_plot)

# Save the final combined plot
ggsave("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition_3.png", final_plot, width = 12, height = 8)


# Create gif from the noise intuition plots using magick package
library(magick)

# List of image files
image_files <- paste0("experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition_", 0:3, ".png")

# Read all images with better quality settings
images <- image_read(image_files)

# Remove transparency and set white background
images <- image_convert(images, colorspace = "sRGB")
images <- image_background(images, "white")

# Combine images into animation with higher quality
animation <- image_join(images)
animation <- image_animate(animation, fps = 0.5, optimize = TRUE)

# Save with higher quality settings
image_write(animation, "experiments/experiment_analysis/dt_es_simulation_study/plots/noise_intuition.gif", quality = 100)

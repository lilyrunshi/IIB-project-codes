# Helper functions for summarising DSC outputs that include a noise sweep.
#
# The tibble returned by dscquery() stores module outputs as character
# columns.  To compute numeric summaries you need to coerce the error metrics
# and noise level into numeric form and tell the summary statistics to drop
# any missing values.  These helpers wrap up those steps so that your R
# session stays tidy and reproducible.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})

#' Prepare the DSC output for numeric summaries.
#'
#' @param dscout A tibble produced by dscquery().
#' @return A tibble with numeric columns ready for aggregation.
normalize_noise_results <- function(dscout) {
  dscout %>%
    mutate(
      simulate.noise_std = parse_number(simulate.noise_std),
      rmse.error = parse_number(rmse.error),
      mae.error = parse_number(mae.error)
    )
}

#' Summarise the RMSE and MAE distribution for each noise level and analysis module.
#'
#' @param dscout A tibble from dscquery().
#' @return A tibble of grouped summary statistics.
summarise_noise_performance <- function(dscout) {
  normalize_noise_results(dscout) %>%
    group_by(simulate.noise_std, analyze) %>%
    summarise(
      rmse_mean = mean(rmse.error, na.rm = TRUE),
      rmse_sd = sd(rmse.error, na.rm = TRUE),
      mae_mean = mean(mae.error, na.rm = TRUE),
      mae_sd = sd(mae.error, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(simulate.noise_std, analyze)
}

#' Convert the grouped summary table into a metric-oriented wide format.
#'
#' @param summary_tbl Output from summarise_noise_performance().
#' @return A tibble with mean/SD columns for each metric.
pivot_noise_summary <- function(summary_tbl) {
  summary_tbl %>%
    pivot_longer(
      cols = c(rmse_mean, rmse_sd, mae_mean, mae_sd),
      names_to = c("metric", ".value"),
      names_pattern = "(.*)_(mean|sd)"
    ) %>%
    arrange(metric, simulate.noise_std, analyze)
}

#' Plot the score distribution as a function of the simulated noise level.
#'
#' @param dscout A tibble from dscquery().
#' @return A ggplot object showing boxplots with mean overlays.
plot_noise_performance <- function(dscout) {
  normalize_noise_results(dscout) %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value"
    ) %>%
    ggplot(aes(x = factor(simulate.noise_std), y = value, fill = analyze)) +
    geom_boxplot(alpha = 0.6, position = position_dodge(width = 0.9), na.rm = TRUE) +
    stat_summary(
      fun = mean,
      geom = "point",
      position = position_dodge(width = 0.9),
      shape = 21,
      size = 2,
      colour = "black"
    ) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(
      x = "Noise standard deviation",
      y = "Error",
      fill = "Model",
      title = "Model performance across simulated noise levels"
    ) +
    theme_minimal()
}

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

coerce_numeric <- function(x) {
  if (is.numeric(x)) {
    return(x)
  }
  parse_number(as.character(x))
}

#' Prepare the DSC output for numeric summaries.
#'
#' @param dscout A tibble produced by dscquery().
#' @return A tibble with numeric columns ready for aggregation.
normalize_noise_results <- function(dscout) {
  dscout %>%
    mutate(across(
      c(simulate.noise_std, rmse.error, mae.error),
      coerce_numeric
    ))
}

#' Prepare the DSC output for sparsity-based summaries.
#'
#' @param dscout A tibble produced by dscquery().
#' @return A tibble with numeric columns ready for aggregation.
normalize_sparsity_results <- function(dscout) {
  dscout %>%
    mutate(across(
      c(simulate.sparsity_prob, rmse.error, mae.error),
      coerce_numeric
    ))
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

#' Plot score distributions per noise level and aggregate trends.
#'
#' @param dscout A tibble from dscquery().
#' @return A list containing individual boxplots for every
#'   (noise level, metric) pair and a summary plot of the mean errors across
#'   noise levels.
plot_noise_performance <- function(dscout) {
  tidy_scores <- normalize_noise_results(dscout) %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value"
    )

  per_noise_groups <- tidy_scores %>%
    group_by(simulate.noise_std, metric)

  per_noise_keys <- group_keys(per_noise_groups)

  per_noise_plots <- group_map(
    per_noise_groups,
    function(df, key) {
      metric_label <- toupper(gsub("\\.error$", "", key$metric))

      ggplot(
        df,
        aes(x = analyze, y = value, fill = analyze)
      ) +
        geom_boxplot(alpha = 0.6, na.rm = TRUE) +
        stat_summary(
          fun = mean,
          geom = "point",
          shape = 21,
          size = 2,
          colour = "black"
        ) +
        labs(
          title = sprintf("%s | noise = %s", metric_label, key$simulate.noise_std),
          x = "Analysis module",
          y = "Error",
          fill = "Model"
        ) +
        theme_minimal()
    }
  )

  names(per_noise_plots) <- sprintf(
    "%s_noise_%s",
    per_noise_keys$metric,
    per_noise_keys$simulate.noise_std
  )

  trend_data <- tidy_scores %>%
    group_by(simulate.noise_std, analyze, metric) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")

  average_plot <- ggplot(
    trend_data,
    aes(x = simulate.noise_std, y = mean_value, colour = analyze)
  ) +
    geom_line() +
    geom_point(size = 2) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(
      title = "Mean error across noise levels",
      x = "Noise standard deviation",
      y = "Mean error",
      colour = "Model"
    ) +
    theme_minimal()

  list(individual = per_noise_plots, average = average_plot)
}

#' Summarise the RMSE and MAE distribution for each sparsity level and analysis module.
#'
#' @param dscout A tibble from dscquery().
#' @return A tibble of grouped summary statistics.
summarise_sparsity_performance <- function(dscout) {
  normalize_sparsity_results(dscout) %>%
    group_by(simulate.sparsity_prob, analyze) %>%
    summarise(
      rmse_mean = mean(rmse.error, na.rm = TRUE),
      rmse_sd = sd(rmse.error, na.rm = TRUE),
      mae_mean = mean(mae.error, na.rm = TRUE),
      mae_sd = sd(mae.error, na.rm = TRUE),
      .groups = "drop",
    ) %>%
    arrange(simulate.sparsity_prob, analyze)
}

#' Convert the grouped sparsity summary table into a metric-oriented wide format.
#'
#' @param summary_tbl Output from summarise_sparsity_performance().
#' @return A tibble with mean/SD columns for each metric.
pivot_sparsity_summary <- function(summary_tbl) {
  summary_tbl %>%
    pivot_longer(
      cols = c(rmse_mean, rmse_sd, mae_mean, mae_sd),
      names_to = c("metric", ".value"),
      names_pattern = "(.*)_(mean|sd)",
    ) %>%
    arrange(metric, simulate.sparsity_prob, analyze)
}

#' Plot score distributions per sparsity level and aggregate trends.
#'
#' @param dscout A tibble from dscquery().
#' @return A list containing individual boxplots for every
#'   (sparsity level, metric) pair and a summary plot of the mean errors across
#'   sparsity levels.
plot_sparsity_performance <- function(dscout) {
  tidy_scores <- normalize_sparsity_results(dscout) %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value",
    )

  per_sparsity_groups <- tidy_scores %>%
    group_by(simulate.sparsity_prob, metric)

  per_sparsity_keys <- group_keys(per_sparsity_groups)

  per_sparsity_plots <- group_map(
    per_sparsity_groups,
    function(df, key) {
      metric_label <- toupper(gsub("\\.error$", "", key$metric))

      ggplot(
        df,
        aes(x = analyze, y = value, fill = analyze),
      ) +
        geom_boxplot(alpha = 0.6, na.rm = TRUE) +
        stat_summary(
          fun = mean,
          geom = "point",
          shape = 21,
          size = 2,
          colour = "black",
        ) +
        labs(
          title = sprintf(
            "%s | sparsity = %s",
            metric_label,
            key$simulate.sparsity_prob,
          ),
          x = "Analysis module",
          y = "Error",
          fill = "Model",
        ) +
        theme_minimal()
    }
  )

  names(per_sparsity_plots) <- sprintf(
    "%s_sparsity_%s",
    per_sparsity_keys$metric,
    per_sparsity_keys$simulate.sparsity_prob,
  )

  trend_data <- tidy_scores %>%
    group_by(simulate.sparsity_prob, analyze, metric) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")

  average_plot <- ggplot(
    trend_data,
    aes(x = simulate.sparsity_prob, y = mean_value, colour = analyze),
  ) +
    geom_line() +
    geom_point(size = 2) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(
      title = "Mean error across sparsity levels",
      x = "Latent sparsity probability",
      y = "Mean error",
      colour = "Model",
    ) +
    theme_minimal()

  list(individual = per_sparsity_plots, average = average_plot)
}

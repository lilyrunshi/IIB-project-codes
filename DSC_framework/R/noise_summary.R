# Helper functions for summarising DSC outputs that include a noise sweep.
#
# The tibble returned by dscquery() stores module outputs as character
# columns.  To compute numeric summaries you need to coerce the error metrics
# and noise level into numeric form and tell the summary statistics to drop
# any missing values.  These helpers wrap up those steps so that your R
# session stays tidy and reproducible.

suppressPackageStartupMessages({
  library(dscrutils)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})

pause_and_store_plots <- function(
    plot_list,
    average_plot,
    output_dir,
    pause_seconds,
    display_plots,
    average_basename) {
  if (is.null(pause_seconds) || is.na(pause_seconds)) {
    pause_seconds <- 0
  }
  pause_seconds <- max(0, pause_seconds)
  display_plots <- isTRUE(display_plots)

  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  for (plot_name in names(plot_list)) {
    plot_obj <- plot_list[[plot_name]]
    if (display_plots) {
      print(plot_obj)
    }
    if (!is.null(output_dir)) {
      ggsave(
        filename = file.path(output_dir, paste0(plot_name, ".png")),
        plot = plot_obj,
        width = 8,
        height = 5,
        dpi = 300
      )
    }
    if (pause_seconds > 0) {
      Sys.sleep(pause_seconds)
    }
  }

  if (display_plots) {
    print(average_plot)
  }
  if (!is.null(output_dir)) {
    ggsave(
      filename = file.path(output_dir, paste0(average_basename, ".png")),
      plot = average_plot,
      width = 8,
      height = 5,
      dpi = 300
    )
  }
  if (pause_seconds > 0) {
    Sys.sleep(pause_seconds)
  }
}

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

#' Prepare the DSC output for joint noise / sparsity summaries.
#'
#' @param dscout A tibble produced by dscquery().
#' @return A tibble with numeric columns ready for aggregation.
normalize_noise_sparsity_results <- function(dscout) {
  dscout %>%
    mutate(across(
      c(simulate.noise_std, simulate.sparsity_prob, rmse.error, mae.error),
      coerce_numeric
    ))
}

require_dscquery <- function() {
  if (!requireNamespace("dscrutils", quietly = TRUE)) {
    stop(
      paste0(
        "Could not load dscrutils::dscquery(). Install the 'dscrutils' package",
        " or provide the DSC table via the `dsc_table` argument."
      ),
      call. = FALSE
    )
  }
  tryCatch(
    getExportedValue("dscrutils", "dscquery"),
    error = function(e) {
      stop(
        paste0(
          "Could not load dscrutils::dscquery(). Install the 'dscrutils' package",
          " or provide the DSC table via the `dsc_table` argument."
        ),
        call. = FALSE
      )
    }
  )
}

#' Query DSC results for noise / sparsity analysis.
#'
#' @param dsc_path Path to the DSC result directory.
#' @param modules Optional vector of module names to include.
#' @param metrics Character vector of metrics (without the `.error` suffix).
#' @param extra_targets Additional target columns to request from dscquery().
#' @return Tibble with the requested DSC outputs.
query_noise_sparsity_results <- function(
    dsc_path = ".",
    modules = NULL,
    metrics = c("rmse", "mae"),
    extra_targets = NULL) {
  dscquery <- require_dscquery()
  metric_targets <- paste0(metrics, ".error")
  targets <- unique(c(
    "simulate.noise_std",
    "simulate.sparsity_prob",
    metric_targets,
    extra_targets
  ))

  results <- dscquery(dsc.outdir = dsc_path, targets = targets)
  results <- as_tibble(results)

  if (!"analysis" %in% names(results)) {
    if ("module" %in% names(results)) {
      results <- mutate(results, analysis = .data$module)
    } else if ("method" %in% names(results)) {
      results <- mutate(results, analysis = .data$method)
    } else {
      warning(
        "Could not find an analysis column in DSC output; synthesising from row names."
      )
      results <- mutate(results, analysis = sprintf("analysis_%s", seq_len(n())))
    }
  }

  if (!is.null(modules)) {
    module_column <- intersect(c("analysis", "module", "method"), names(results))
    if (length(module_column) > 0) {
      results <- results %>%
        filter(.data[[module_column[[1]]]] %in% modules)
    } else {
      warning(
        paste(
          "`modules` argument supplied but DSC results do not contain a recognised",
          "module column; returning all analyses."
        )
      )
    }
  }

  results
}

#' Convert error metrics to a tidy long format for plotting.
#'
#' @param dscout Tibble with `rmse.error` / `mae.error` columns.
#' @return Long-format tibble containing one row per metric.
metrics_long <- function(dscout) {
  dscout %>%
    pivot_longer(
      cols = ends_with(".error"),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = sub("\\.error$", "", .data$metric),
      metric = factor(.data$metric, levels = c("rmse", "mae"))
    )
}

summarise_noise_metrics <- function(dscout) {
  dscout %>%
    normalize_noise_results() %>%
    metrics_long() %>%
    group_by(.data$analysis, .data$metric, .data$simulate.noise_std) %>%
    summarise(
      mean = mean(.data$value, na.rm = TRUE),
      median = median(.data$value, na.rm = TRUE),
      sd = sd(.data$value, na.rm = TRUE),
      q25 = quantile(.data$value, probs = 0.25, na.rm = TRUE),
      q75 = quantile(.data$value, probs = 0.75, na.rm = TRUE),
      .groups = "drop"
    )
}

summarise_sparsity_metrics <- function(dscout) {
  dscout %>%
    normalize_sparsity_results() %>%
    metrics_long() %>%
    group_by(.data$analysis, .data$metric, .data$simulate.sparsity_prob) %>%
    summarise(
      mean = mean(.data$value, na.rm = TRUE),
      median = median(.data$value, na.rm = TRUE),
      sd = sd(.data$value, na.rm = TRUE),
      q25 = quantile(.data$value, probs = 0.25, na.rm = TRUE),
      q75 = quantile(.data$value, probs = 0.75, na.rm = TRUE),
      .groups = "drop"
    )
}

summarise_noise_sparsity_metrics <- function(dscout) {
  dscout %>%
    normalize_noise_sparsity_results() %>%
    metrics_long() %>%
    group_by(
      .data$analysis,
      .data$metric,
      .data$simulate.noise_std,
      .data$simulate.sparsity_prob
    ) %>%
    summarise(
      mean = mean(.data$value, na.rm = TRUE),
      median = median(.data$value, na.rm = TRUE),
      sd = sd(.data$value, na.rm = TRUE),
      .groups = "drop"
    )
}

plot_noise_trends <- function(noise_summary) {
  ggplot(
    noise_summary,
    aes(x = .data$simulate.noise_std, y = .data$mean, colour = .data$analysis)
  ) +
    geom_line() +
    geom_point() +
    facet_wrap(~metric, scales = "free_y") +
    labs(
      x = "Noise standard deviation",
      y = "Mean percentage error",
      colour = "Analysis",
      title = "Performance trend across noise levels"
    ) +
    theme_minimal()
}

plot_noise_boxplots <- function(dscout) {
  metrics_long(normalize_noise_results(dscout)) %>%
    ggplot(
      aes(
        x = factor(.data$simulate.noise_std),
        y = .data$value,
        fill = .data$analysis
      )
    ) +
    geom_boxplot(outlier.shape = NA, position = position_dodge(width = 0.75)) +
    labs(
      x = "Noise standard deviation",
      y = "Percentage error",
      fill = "Analysis",
      title = "Distribution of errors per noise level"
    ) +
    facet_wrap(~metric, scales = "free_y") +
    theme_minimal()
}

plot_sparsity_trends <- function(sparsity_summary) {
  ggplot(
    sparsity_summary,
    aes(
      x = .data$simulate.sparsity_prob,
      y = .data$mean,
      colour = .data$analysis
    )
  ) +
    geom_line() +
    geom_point() +
    facet_wrap(~metric, scales = "free_y") +
    labs(
      x = "Sparsity probability",
      y = "Mean percentage error",
      colour = "Analysis",
      title = "Performance trend across sparsity levels"
    ) +
    theme_minimal()
}

plot_sparsity_boxplots <- function(dscout) {
  metrics_long(normalize_sparsity_results(dscout)) %>%
    ggplot(
      aes(
        x = factor(.data$simulate.sparsity_prob),
        y = .data$value,
        fill = .data$analysis
      )
    ) +
    geom_boxplot(outlier.shape = NA, position = position_dodge(width = 0.75)) +
    labs(
      x = "Sparsity probability",
      y = "Percentage error",
      fill = "Analysis",
      title = "Distribution of errors per sparsity level"
    ) +
    facet_wrap(~metric, scales = "free_y") +
    theme_minimal()
}

plot_noise_sparsity_heatmap <- function(joint_summary) {
  ggplot(
    joint_summary,
    aes(
      x = .data$simulate.sparsity_prob,
      y = .data$simulate.noise_std,
      fill = .data$mean
    )
  ) +
    geom_tile() +
    facet_grid(metric ~ analysis) +
    scale_fill_viridis_c(option = "plasma", direction = -1, na.value = "grey80") +
    labs(
      x = "Sparsity probability",
      y = "Noise standard deviation",
      fill = "Mean percentage error",
      title = "Joint noise / sparsity performance heatmap"
    ) +
    theme_minimal()
}

plot_average_performance <- function(joint_summary) {
  joint_summary %>%
    group_by(.data$analysis, .data$metric) %>%
    summarise(
      overall_mean = mean(.data$mean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    ggplot(aes(x = .data$analysis, y = .data$overall_mean, fill = .data$metric)) +
    geom_col(position = position_dodge(width = 0.75)) +
    labs(
      x = "Analysis module",
      y = "Average mean percentage error",
      fill = "Metric",
      title = "Average performance across noise and sparsity grid"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Run the full noise / sparsity analysis workflow.
#'
#' @param dsc_path Path to the DSC results directory.
#' @param dsc_table Optional pre-loaded DSC tibble (skips querying if provided).
#' @param modules Optional vector restricting the analyses considered.
#' @param metrics Metrics to analyse (defaults to RMSE and MAE).
#' @param output_dir Directory where PNGs should be written (NULL to skip).
#' @param pause_seconds Seconds to pause between plot renders (useful for knitting).
#' @param display_plots Whether to print plots to the active device.
#' @return A list containing the queried data, summaries, and ggplot objects.
run_noise_sparsity_analysis <- function(
    dsc_path = ".",
    dsc_table = NULL,
    modules = NULL,
    metrics = c("rmse", "mae"),
    output_dir = file.path(dsc_path, "plot_outputs"),
    pause_seconds = 0,
    display_plots = interactive()) {
  if (is.null(dsc_table)) {
    dsc_table <- query_noise_sparsity_results(
      dsc_path = dsc_path,
      modules = modules,
      metrics = metrics
    )
  } else {
    dsc_table <- as_tibble(dsc_table)
  }

  if (!all(paste0(metrics, ".error") %in% names(dsc_table))) {
    stop(
      "Not all requested metrics were found in the DSC table.",
      call. = FALSE
    )
  }

  noise_summary <- summarise_noise_metrics(dsc_table)
  sparsity_summary <- summarise_sparsity_metrics(dsc_table)
  joint_summary <- summarise_noise_sparsity_metrics(dsc_table)

  plot_objects <- list(
    noise_trends = plot_noise_trends(noise_summary),
    noise_boxplots = plot_noise_boxplots(dsc_table),
    sparsity_trends = plot_sparsity_trends(sparsity_summary),
    sparsity_boxplots = plot_sparsity_boxplots(dsc_table),
    noise_sparsity_heatmap = plot_noise_sparsity_heatmap(joint_summary)
  )
  average_plot <- plot_average_performance(joint_summary)

  pause_and_store_plots(
    plot_list = plot_objects,
    average_plot = average_plot,
    output_dir = output_dir,
    pause_seconds = pause_seconds,
    display_plots = display_plots,
    average_basename = "average_performance"
  )

  list(
    dsc_table = dsc_table,
    noise_summary = noise_summary,
    sparsity_summary = sparsity_summary,
    joint_summary = joint_summary,
    plots = c(plot_objects, list(average_performance = average_plot))
  )
}

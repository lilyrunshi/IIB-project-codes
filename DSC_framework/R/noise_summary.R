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

most_common_value <- function(values) {
  values <- values[!is.na(values)]
  if (length(values) == 0) {
    return(NA_real_)
  }

  freq_table <- sort(table(values), decreasing = TRUE)
  as.numeric(names(freq_table)[1])
}

annotate_joint_sweep <- function(dscout) {
  tidy <- normalize_noise_sparsity_results(dscout)

  baseline_noise <- most_common_value(tidy$simulate.noise_std)
  baseline_sparsity <- most_common_value(tidy$simulate.sparsity_prob)

  annotated <- tidy %>%
    mutate(
      sweep_factor = case_when(
        !is.na(simulate.noise_std) && !is.na(baseline_noise) &&
          !is.na(simulate.sparsity_prob) && !is.na(baseline_sparsity) &&
          !near(simulate.noise_std, baseline_noise) &&
          !near(simulate.sparsity_prob, baseline_sparsity) ~ "multiple",
        !is.na(simulate.noise_std) && !is.na(baseline_noise) &&
          !near(simulate.noise_std, baseline_noise) ~ "noise",
        !is.na(simulate.sparsity_prob) && !is.na(baseline_sparsity) &&
          !near(simulate.sparsity_prob, baseline_sparsity) ~ "sparsity",
        TRUE ~ "baseline"
      ),
      sweep_value = case_when(
        sweep_factor == "noise" ~ simulate.noise_std,
        sweep_factor == "sparsity" ~ simulate.sparsity_prob,
        TRUE ~ NA_real_
      ),
      baseline_noise = baseline_noise,
      baseline_sparsity = baseline_sparsity
    )

  attr(annotated, "baseline_noise") <- baseline_noise
  attr(annotated, "baseline_sparsity") <- baseline_sparsity
  annotated
}

check_single_factor_design <- function(annotated) {
  baseline_noise <- attr(annotated, "baseline_noise")
  baseline_sparsity <- attr(annotated, "baseline_sparsity")

  conflicting_rows <- annotated %>%
    filter(
      sweep_factor == "multiple" || (
        !is.na(baseline_noise) && !is.na(baseline_sparsity) &&
          !is.na(simulate.noise_std) && !is.na(simulate.sparsity_prob) &&
          !near(simulate.noise_std, baseline_noise) &&
          !near(simulate.sparsity_prob, baseline_sparsity)
      )
    )

  if (nrow(conflicting_rows) > 0) {
    stop(paste0(
      "Detected runs where both noise and sparsity deviate from the baseline. ",
      "The helper functions assume the DSC varies one factor at a time."
    ))
  }

  annotated
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
#' @param pause_seconds Number of seconds to wait after printing each plot.
#'   Defaults to 1 second.
#' @param output_dir Directory where plots should be written.  The directory
#'   will be created if it does not exist.  Set to `NULL` to skip saving.
#' @param display_plots Should the plots be printed as they are generated?
#'   Defaults to `interactive()`.
#' @return A list containing individual boxplots for every (noise level, metric)
#'   pair and a summary plot of the mean errors across noise levels.
plot_noise_performance <- function(
    dscout,
    pause_seconds = 1,
    output_dir = file.path("plot_outputs", "noise"),
    display_plots = interactive()) {
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

  pause_and_store_plots(
    per_noise_plots,
    average_plot,
    output_dir,
    pause_seconds,
    display_plots,
    average_basename = "average_noise"
  )

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
#' @param pause_seconds Number of seconds to wait after printing each plot.
#'   Defaults to 1 second.
#' @param output_dir Directory where plots should be written.  The directory
#'   will be created if it does not exist.  Set to `NULL` to skip saving.
#' @param display_plots Should the plots be printed as they are generated?
#'   Defaults to `interactive()`.
#' @return A list containing individual boxplots for every (sparsity level,
#'   metric) pair and a summary plot of the mean errors across sparsity levels.
plot_sparsity_performance <- function(
    dscout,
    pause_seconds = 1,
    output_dir = file.path("plot_outputs", "sparsity"),
    display_plots = interactive()) {
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

  pause_and_store_plots(
    per_sparsity_plots,
    average_plot,
    output_dir,
    pause_seconds,
    display_plots,
    average_basename = "average_sparsity"
  )

  list(individual = per_sparsity_plots, average = average_plot)
}

#' Summarise the RMSE and MAE distribution for noise and sparsity sweeps.
#' 
#' @param dscout A tibble from dscquery().
#' @return A tibble of grouped summary statistics with the varied factor recorded.
summarise_noise_sparsity_performance <- function(dscout) {
  annotated <- dscout %>%
    annotate_joint_sweep() %>%
    check_single_factor_design()

  noise_summary <- annotated %>%
    filter(sweep_factor != "sparsity") %>%
    summarise_noise_performance() %>%
    rename(sweep_value = simulate.noise_std) %>%
    mutate(sweep_factor = "noise")

  sparsity_summary <- annotated %>%
    filter(sweep_factor != "noise") %>%
    summarise_sparsity_performance() %>%
    rename(sweep_value = simulate.sparsity_prob) %>%
    mutate(sweep_factor = "sparsity")

  bind_rows(noise_summary, sparsity_summary) %>%
    relocate(sweep_factor, sweep_value) %>%
    arrange(sweep_factor, sweep_value, analyze)
}

#' Convert the grouped summary table into a metric-oriented wide format.
#'
#' @param summary_tbl Output from summarise_noise_sparsity_performance().
#' @return A tibble with mean/SD columns for each metric.
pivot_noise_sparsity_summary <- function(summary_tbl) {
  summary_tbl %>%
    pivot_longer(
      cols = c(rmse_mean, rmse_sd, mae_mean, mae_sd),
      names_to = c("metric", ".value"),
      names_pattern = "(.*)_(mean|sd)"
    ) %>%
    arrange(metric, sweep_factor, sweep_value, analyze)
}

#' Plot score distributions for each sweep factor and aggregate trends.
#'
#' @param dscout A tibble from dscquery().
#' @param pause_seconds Number of seconds to wait after printing each plot.
#'   Defaults to 1 second.
#' @param output_dir Directory where plots should be written.  The directory
#'   will be created if it does not exist.  Set to `NULL` to skip saving.
#' @param display_plots Should the plots be printed as they are generated?
#'   Defaults to `interactive()`.
#' @return A list containing per-factor boxplots (with baseline reference) and an
#'   overview plot aggregating across analysis modules.
plot_noise_sparsity_performance <- function(
    dscout,
    pause_seconds = 1,
    output_dir = file.path("plot_outputs", "noise_sparsity"),
    display_plots = interactive()) {
  annotated <- dscout %>%
    annotate_joint_sweep() %>%
    check_single_factor_design()

  tidy_scores <- annotated %>%
    filter(sweep_factor != "baseline") %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value"
    )

  baseline_summary <- annotated %>%
    filter(sweep_factor == "baseline") %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value"
    ) %>%
    group_by(analyze, metric) %>%
    summarise(baseline_mean = mean(value, na.rm = TRUE), .groups = "drop")

  per_factor_groups <- tidy_scores %>%
    group_by(sweep_factor, metric)

  per_factor_keys <- group_keys(per_factor_groups)

  per_factor_plots <- group_map(
    per_factor_groups,
    function(df, key) {
      metric_label <- toupper(gsub("\\.error$", "", key$metric))
      baseline_lines <- baseline_summary %>%
        filter(metric == key$metric)
      level_order <- sort(unique(df$sweep_value))
      level_order <- level_order[!is.na(level_order)]

      ggplot(
        df,
        aes(x = factor(sweep_value, levels = level_order), y = value, fill = analyze)
      ) +
        geom_boxplot(alpha = 0.6, na.rm = TRUE) +
        stat_summary(
          fun = mean,
          geom = "point",
          shape = 21,
          size = 2,
          colour = "black"
        ) +
        geom_hline(
          data = baseline_lines,
          aes(yintercept = baseline_mean, colour = analyze),
          linetype = "dashed",
          inherit.aes = FALSE
        ) +
        scale_colour_discrete(guide = "none") +
        labs(
          title = sprintf(
            "%s | %s sweep",
            metric_label,
            key$sweep_factor
          ),
          x = ifelse(
            key$sweep_factor == "noise",
            "Noise standard deviation",
            "Latent sparsity probability"
          ),
          y = "Error",
          fill = "Model"
        ) +
        theme_minimal()
    }
  )

  names(per_factor_plots) <- sprintf(
    "%s_%s_sweep",
    per_factor_keys$metric,
    per_factor_keys$sweep_factor
  )

  trend_data <- tidy_scores %>%
    group_by(sweep_factor, sweep_value, analyze, metric) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")

  average_plot <- ggplot(
    trend_data,
    aes(x = sweep_value, y = mean_value, colour = analyze)
  ) +
    geom_line() +
    geom_point(size = 2) +
    facet_grid(
      metric ~ sweep_factor,
      scales = "free_y",
      labeller = labeller(
        sweep_factor = c(noise = "Noise standard deviation", sparsity = "Latent sparsity probability"),
        metric = function(x) toupper(gsub("\\.error$", "", x))
      )
    ) +
    labs(
      title = "Mean error across noise and sparsity sweeps",
      x = "Sweep level",
      y = "Mean error",
      colour = "Model"
    ) +
    theme_minimal()

  pause_and_store_plots(
    per_factor_plots,
    average_plot,
    output_dir,
    pause_seconds,
    display_plots,
    average_basename = "average_noise_sparsity"
  )

  list(boxplots = per_factor_plots, average = average_plot)
}

#' Run a full DSC noise/sparsity analysis workflow.
#'
#' This helper collects the common steps discussed in the notebook workflow:
#' load the required packages, query the DSC output, produce tidy summaries and
#' generate the standard plots.  It returns a list so you can inspect both the
#' summaries and the `ggplot` objects programmatically.
#'
#' @param dsc_path Path to the DSC run directory (the folder that contains the
#'   `dsc-result.rds` file).
#' @param pause_seconds Seconds to wait between plot renders.  Defaults to 0 so
#'   scripted runs finish quickly.
#' @param output_root Base directory where the noise/sparsity plots should be
#'   written.  Subdirectories for each analysis type will be created
#'   automatically.
#' @param display_plots Should plots be printed while the workflow runs?
#'   Defaults to `interactive()`.
#' @return A list with the raw DSC tibble, per-factor summaries and the plot
#'   objects produced during the workflow.
run_noise_sparsity_analysis <- function(
    dsc_path,
    pause_seconds = 0,
    output_root = "plot_outputs",
    display_plots = interactive()) {
  message("Loading DSC outputs from: ", dsc_path)
  dscout <- dscrutils::dscquery(
    dsc.outdir = dsc_path,
    targets = c(
      "simulate.noise_std",
      "simulate.sparsity_prob",
      "analyze",
      "rmse.error",
      "mae.error"
    )
  )

  if (!inherits(dscout, "tbl_df")) {
    dscout <- dplyr::as_tibble(dscout)
  }

  message("Summarising noise sweep performance ...")
  noise_summary <- summarise_noise_performance(dscout)
  message("Summarising sparsity sweep performance ...")
  sparsity_summary <- summarise_sparsity_performance(dscout)
  message("Summarising combined sweep performance ...")
  joint_summary <- summarise_noise_sparsity_performance(dscout)

  message("Rendering plots ...")
  noise_plots <- plot_noise_performance(
    dscout,
    pause_seconds = pause_seconds,
    output_dir = file.path(output_root, "noise"),
    display_plots = display_plots
  )
  sparsity_plots <- plot_sparsity_performance(
    dscout,
    pause_seconds = pause_seconds,
    output_dir = file.path(output_root, "sparsity"),
    display_plots = display_plots
  )
  joint_plots <- plot_noise_sparsity_performance(
    dscout,
    pause_seconds = pause_seconds,
    output_dir = file.path(output_root, "noise_sparsity"),
    display_plots = display_plots
  )

  list(
    dsc_output = dscout,
    noise_summary = noise_summary,
    sparsity_summary = sparsity_summary,
    joint_summary = joint_summary,
    plots = list(
      noise = noise_plots,
      sparsity = sparsity_plots,
      noise_sparsity = joint_plots
    )
  )
}

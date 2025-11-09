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
        !is.na(simulate.noise_std) & !is.na(baseline_noise) &
          !is.na(simulate.sparsity_prob) & !is.na(baseline_sparsity) &
          !near(simulate.noise_std, baseline_noise) &
          !near(simulate.sparsity_prob, baseline_sparsity) ~ "joint",
        !is.na(simulate.noise_std) & !is.na(baseline_noise) &
          !near(simulate.noise_std, baseline_noise) ~ "noise",
        !is.na(simulate.sparsity_prob) & !is.na(baseline_sparsity) &
          !near(simulate.sparsity_prob, baseline_sparsity) ~ "sparsity",
        TRUE ~ "baseline"
      ),
      sweep_value = case_when(
        sweep_factor == "noise" ~ simulate.noise_std,
        sweep_factor == "sparsity" ~ simulate.sparsity_prob,
        TRUE ~ NA_real_
      ),
      sweep_noise = simulate.noise_std,
      sweep_sparsity = simulate.sparsity_prob,
      baseline_noise = baseline_noise,
      baseline_sparsity = baseline_sparsity
    )

  attr(annotated, "baseline_noise") <- baseline_noise
  attr(annotated, "baseline_sparsity") <- baseline_sparsity
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
      .groups = "drop"
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
      names_pattern = "(.*)_(mean|sd)"
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
      values_to = "value"
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
          title = sprintf(
            "%s | sparsity = %s",
            metric_label,
            key$simulate.sparsity_prob
          ),
          x = "Analysis module",
          y = "Error",
          fill = "Model"
        ) +
        theme_minimal()
    }
  )

  names(per_sparsity_plots) <- sprintf(
    "%s_sparsity_%s",
    per_sparsity_keys$metric,
    per_sparsity_keys$simulate.sparsity_prob
  )

  trend_data <- tidy_scores %>%
    group_by(simulate.sparsity_prob, analyze, metric) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")

  average_plot <- ggplot(
    trend_data,
    aes(x = simulate.sparsity_prob, y = mean_value, colour = analyze)
  ) +
    geom_line() +
    geom_point(size = 2) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(
      title = "Mean error across sparsity levels",
      x = "Latent sparsity probability",
      y = "Mean error",
      colour = "Model"
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
#' @return A list of grouped summary statistics for noise, sparsity and joint sweeps.
summarise_noise_sparsity_performance <- function(dscout) {
  annotated <- annotate_joint_sweep(dscout)

  noise_summary <- annotated %>%
    summarise_noise_performance()

  sparsity_summary <- annotated %>%
    summarise_sparsity_performance()

  joint_summary <- annotated %>%
    normalize_noise_sparsity_results() %>%
    group_by(simulate.noise_std, simulate.sparsity_prob, analyze) %>%
    summarise(
      rmse_mean = mean(rmse.error, na.rm = TRUE),
      rmse_sd = sd(rmse.error, na.rm = TRUE),
      mae_mean = mean(mae.error, na.rm = TRUE),
      mae_sd = sd(mae.error, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(simulate.noise_std, simulate.sparsity_prob, analyze)

  list(
    noise = noise_summary,
    sparsity = sparsity_summary,
    joint = joint_summary
  )
}

#' Convert the grouped summary table into a metric-oriented wide format.
#'
#' @param summary_tbl Output from summarise_noise_sparsity_performance(). Provide one
#'   of the list components (noise, sparsity or joint).
#' @return A tibble with mean/SD columns for each metric. Joint summaries retain both
#'   sweep dimensions.
pivot_noise_sparsity_summary <- function(summary_tbl) {
  summary_tbl %>%
    pivot_longer(
      cols = c(rmse_mean, rmse_sd, mae_mean, mae_sd),
      names_to = c("metric", ".value"),
      names_pattern = "(.*)_(mean|sd)"
    ) %>%
    arrange(
      metric,
      across(matches("simulate\\.")),
      analyze
    )
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
#' @return A list containing per-factor boxplots (with baseline reference), a
#'   heatmap of joint means and a trend plot aggregating across analysis modules.
plot_noise_sparsity_performance <- function(
    dscout,
    pause_seconds = 1,
    output_dir = file.path("plot_outputs", "noise_sparsity"),
    display_plots = interactive()) {
  annotated <- annotate_joint_sweep(dscout)

  baseline_noise <- attr(annotated, "baseline_noise")
  baseline_sparsity <- attr(annotated, "baseline_sparsity")

  tidy_scores <- annotated %>%
    pivot_longer(
      cols = c(rmse.error, mae.error),
      names_to = "metric",
      values_to = "value"
    )

  baseline_summary <- tidy_scores %>%
    filter(
      (is.na(baseline_noise) | near(simulate.noise_std, baseline_noise)),
      (is.na(baseline_sparsity) | near(simulate.sparsity_prob, baseline_sparsity))
    ) %>%
    group_by(analyze, metric) %>%
    summarise(baseline_mean = mean(value, na.rm = TRUE), .groups = "drop")

  per_factor_scores <- bind_rows(
    tidy_scores %>%
      mutate(sweep_factor = "noise", sweep_value = simulate.noise_std) %>%
      filter(!is.na(sweep_value)),
    tidy_scores %>%
      mutate(sweep_factor = "sparsity", sweep_value = simulate.sparsity_prob) %>%
      filter(!is.na(sweep_value))
  )

  per_factor_groups <- per_factor_scores %>%
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

  joint_means <- tidy_scores %>%
    filter(!is.na(simulate.noise_std), !is.na(simulate.sparsity_prob)) %>%
    group_by(simulate.noise_std, simulate.sparsity_prob, analyze, metric) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")

  joint_heatmap <- ggplot(
    joint_means,
    aes(
      x = factor(simulate.noise_std),
      y = factor(simulate.sparsity_prob),
      fill = mean_value
    )
  ) +
    geom_tile(colour = "white") +
    facet_grid(metric ~ analyze) +
    scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
    labs(
      title = "Mean error across noise and sparsity combinations",
      x = "Noise standard deviation",
      y = "Latent sparsity probability",
      fill = "Mean error"
    ) +
    theme_minimal()

  joint_trend <- ggplot(
    joint_means,
    aes(
      x = simulate.noise_std,
      y = mean_value,
      colour = factor(simulate.sparsity_prob)
    )
  ) +
    geom_line() +
    geom_point(size = 2) +
    facet_grid(metric ~ analyze, scales = "free_y") +
    labs(
      title = "Mean error trends across joint sweeps",
      x = "Noise standard deviation",
      y = "Mean error",
      colour = "Sparsity probability"
    ) +
    theme_minimal()

  combined_plots <- c(
    per_factor_plots,
    list(joint_heatmap = joint_heatmap)
  )

  pause_and_store_plots(
    combined_plots,
    joint_trend,
    output_dir,
    pause_seconds,
    display_plots,
    average_basename = "average_noise_sparsity"
  )

  list(
    boxplots = per_factor_plots,
    heatmap = joint_heatmap,
    trend = joint_trend
  )
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
    dsc_path = NULL,
    dsc_table = NULL,
    pause_seconds = 0,
    output_root = "plot_outputs",
    display_plots = interactive()) {
  if (!is.null(dsc_table)) {
    dscout <- dsc_table
  } else {
    if (is.null(dsc_path)) {
      stop(
        "Either provide `dsc_table` with pre-loaded DSC results or specify `dsc_path`.",
        call. = FALSE
      )
    }

    message("Loading DSC outputs from: ", dsc_path)
    query_fun <- tryCatch(
      getExportedValue("dscrutils", "dscquery"),
      error = function(e) {
        stop(
          paste0(
            "Could not load dscrutils::dscquery(). Install the 'dscrutils' package and its ",
            "dependencies or query the DSC results manually and supply them via the `dsc_table` ",
            "argument. Original error: ",
            conditionMessage(e)
          ),
          call. = FALSE
        )
      }
    )

    dscout <- tryCatch(
      query_fun(
        dsc.outdir = dsc_path,
        targets = c(
          "simulate.noise_std",
          "simulate.sparsity_prob",
          "analyze",
          "rmse.error",
          "mae.error"
        )
      ),
      error = function(e) {
        stop(
          paste0(
            "Failed to query DSC results via dscrutils::dscquery(). If the optional 'dsc' ",
            "dependency is unavailable, run the query manually and pass the resulting tibble ",
            "through the `dsc_table` argument. Original error: ",
            conditionMessage(e)
          ),
          call. = FALSE
        )
      }
    )
  }

  

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

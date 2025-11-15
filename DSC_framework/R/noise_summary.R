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

# Helper functions for visualising simulated data alongside model predictions.
#
# These helpers focus on loading the Python pickle outputs produced by the DSC
# pipeline and overlaying the generated observations with the fitted curves from
# each analysis module.

suppressPackageStartupMessages({
  library(ggplot2)
})

#' Ensure that the reticulate package is available for reading Python pickle
#' files. The DSC workflow stores arrays in pickle format, which we load via
#' reticulate so that NumPy arrays are automatically converted to their R
#' counterparts.
#'
#' @return The reticulate namespace.
require_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop(
      paste(
        "The 'reticulate' package is required to read Python pickle files.",
        "Install it via install.packages('reticulate') before calling",
        "plot_prediction_curves()."
      ),
      call. = FALSE
    )
  }
  asNamespace("reticulate")
}

#' Load a Python pickle file and return it as a converted R object.
#'
#' @param path Path to the pickle file.
#' @return A list containing the objects stored in the pickle file.
read_pickle <- function(path) {
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  require_reticulate()

  script <- sprintf(
    paste(
      "import pickle",
      "with open(r'''%s''', 'rb') as fh:",
      "    obj = pickle.load(fh)",
      sep = "\n"
    ),
    path
  )

  env <- reticulate$py_run_string(script, convert = TRUE, local = TRUE)
  env$obj
}

#' Extract a numeric parameter from the DSC debug script.
#'
#' The DSC runtime records the executed Python script as a string. We parse
#' simple assignments such as ``noise_std = 0.5`` to recover metadata for use in
#' plot titles.
#'
#' @param script Character scalar containing the recorded Python script.
#' @param name Parameter name to extract.
#' @return Numeric value if the parameter is present, otherwise ``NA_real_``.
extract_script_param <- function(script, name) {
  if (is.null(script) || is.na(script) || !nzchar(script)) {
    return(NA_real_)
  }
  pattern <- paste0(name, "\\s*=\\s*([-0-9.eE]+)")
  match <- regexec(pattern, script, perl = TRUE)
  captured <- regmatches(script, match)
  if (length(captured) == 0L || length(captured[[1]]) < 2L) {
    return(NA_real_)
  }
  as.numeric(captured[[1]][2])
}

#' Format a numeric value for use in file names.
format_for_filename <- function(x) {
  if (is.na(x)) {
    return("unknown")
  }
  gsub("\\.", "_", format(x, trim = TRUE, scientific = FALSE))
}

#' Collect prediction curves for every simulation replicate.
#'
#' @param dsc_path Path to the DSC output directory (default ``"dsc_result"``).
#' @param feature_index Column from ``simulate.x`` to use as the x-axis when the
#'   simulated covariates are stored as a matrix. Defaults to the first column.
#' @param pause_seconds Optional pause between rendering/saving successive plots.
#' @param output_dir Directory used to store generated figures. Set to ``NULL``
#'   to skip saving.
#' @param display_plots Should the plots be printed as they are generated?
#'   Defaults to ``interactive()``.
#' @param replicates Optional integer vector specifying which DSC replicates to
#'   include. When ``NULL`` all replicates are plotted.
#' @return A named list containing the assembled plotting data and ``ggplot``
#'   objects for each replicate that could be processed.
plot_prediction_curves <- function(
    dsc_path = "dsc_result",
    feature_index = 1L,
    pause_seconds = 0,
    output_dir = file.path("plot_outputs", "predictions"),
    display_plots = interactive(),
    replicates = NULL) {
  reticulate <- require_reticulate()
  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  pause_seconds <- max(0, pause_seconds %||% 0)
  display_plots <- isTRUE(display_plots)

  dsc_path <- normalizePath(dsc_path, winslash = "/", mustWork = TRUE)
  simulate_dir <- file.path(dsc_path, "model3_simulate")
  if (!dir.exists(simulate_dir)) {
    stop("Could not locate the 'model3_simulate' directory inside ", dsc_path, call. = FALSE)
  }

  analyze_dirs <- list.dirs(dsc_path, recursive = FALSE, full.names = FALSE)
  analyze_dirs <- analyze_dirs[analyze_dirs != "model3_simulate"]
  analyze_dirs <- setdiff(analyze_dirs, c("rmse", "mae"))
  analyze_dirs <- analyze_dirs[file.info(file.path(dsc_path, analyze_dirs))$isdir]
  analyze_dirs <- analyze_dirs[vapply(
    analyze_dirs,
    function(dir) {
      any(grepl("\\.pkl$", list.files(file.path(dsc_path, dir), full.names = FALSE)))
    },
    logical(1)
  )]
  if (length(analyze_dirs) == 0L) {
    stop("No analysis module outputs were found inside ", dsc_path, call. = FALSE)
  }

  sim_files <- list.files(simulate_dir, pattern = "\\.pkl$", full.names = TRUE)
  if (length(sim_files) == 0L) {
    stop("No simulation outputs were found inside ", simulate_dir, call. = FALSE)
  }

  plot_results <- list()

  for (sim_path in sim_files) {
    sim_obj <- tryCatch(read_pickle(sim_path), error = identity)
    if (inherits(sim_obj, "error")) {
      warning("Failed to read ", sim_path, ": ", conditionMessage(sim_obj))
      next
    }

    y <- sim_obj$y
    x <- sim_obj$x
    meta <- sim_obj$DSC_DEBUG

    if (is.null(y) || length(y) == 0L) {
      warning("Simulation output at ", sim_path, " did not contain a 'y' vector; skipping.")
      next
    }

    if (is.matrix(x)) {
      if (feature_index > ncol(x)) {
        warning(
          sprintf(
            "feature_index (%d) exceeds the number of columns (%d) in '%s'; skipping replicate.",
            feature_index,
            ncol(x),
            basename(sim_path)
          )
        )
        next
      }
      x_vals <- x[, feature_index]
    } else if (is.null(x)) {
      x_vals <- seq_along(y)
    } else {
      x_vals <- as.numeric(x)
    }

    if (length(x_vals) != length(y)) {
      warning(
        sprintf(
          "Length mismatch between x (length %d) and y (length %d) in '%s'; using sample index instead.",
          length(x_vals),
          length(y),
          basename(sim_path)
        )
      )
      x_vals <- seq_along(y)
    }

    replicate_id <- if (!is.null(meta$replicate)) meta$replicate else NA_integer_
    if (!is.null(replicates) && !is.na(replicate_id) && !(replicate_id %in% replicates)) {
      next
    }

    script <- if (!is.null(meta$script)) meta$script else ""
    noise <- extract_script_param(script, "noise_std")
    sparsity <- extract_script_param(script, "sparsity_prob")

    base_name <- tools::file_path_sans_ext(basename(sim_path))
    plot_key <- sprintf(
      "%s_noise_%s_sparsity_%s_rep_%s",
      base_name,
      format_for_filename(noise),
      format_for_filename(sparsity),
      if (is.na(replicate_id)) "unknown" else replicate_id
    )

    observation_df <- data.frame(
      sample = seq_along(y),
      x_value = as.numeric(x_vals),
      y = as.numeric(y)
    )

    prediction_df <- NULL
    for (module in analyze_dirs) {
      pred_path <- file.path(
        dsc_path,
        module,
        paste0(base_name, "_", module, "_1.pkl")
      )
      if (!file.exists(pred_path)) {
        next
      }

      pred_obj <- tryCatch(read_pickle(pred_path), error = identity)
      if (inherits(pred_obj, "error")) {
        warning("Failed to read ", pred_path, ": ", conditionMessage(pred_obj))
        next
      }
      if (is.null(pred_obj$y_hat)) {
        warning("Model output at ", pred_path, " did not contain 'y_hat'; skipping.")
        next
      }

      if (length(pred_obj$y_hat) != nrow(observation_df)) {
        warning(
          sprintf(
            "Prediction length mismatch for %s (got %d, expected %d); skipping.",
            basename(pred_path),
            length(pred_obj$y_hat),
            nrow(observation_df)
          )
        )
        next
      }

      module_df <- data.frame(
        sample = seq_along(pred_obj$y_hat),
        x_value = observation_df$x_value,
        model = module,
        prediction = as.numeric(pred_obj$y_hat)
      )
      prediction_df <- if (is.null(prediction_df)) module_df else rbind(prediction_df, module_df)
    }

    if (is.null(prediction_df)) {
      warning("No predictions found for replicate ", base_name, "; skipping plot.")
      next
    }

    prediction_df <- prediction_df[order(prediction_df$model, prediction_df$x_value), ]

    plot_title <- sprintf(
      "Replicate %s | noise = %s | sparsity = %s",
      if (is.na(replicate_id)) base_name else replicate_id,
      if (is.na(noise)) "?" else format(noise, digits = 3, trim = TRUE),
      if (is.na(sparsity)) "?" else format(sparsity, digits = 3, trim = TRUE)
    )

    p <- ggplot(observation_df, aes(x = x_value, y = y)) +
      geom_point(colour = "black", alpha = 0.6, size = 1.2) +
      geom_line(
        data = prediction_df,
        aes(x = x_value, y = prediction, colour = model, group = model),
        linewidth = 0.7,
        alpha = 0.9
      ) +
      labs(
        title = plot_title,
        x = "Simulated covariate",
        y = "Response",
        colour = "Model"
      ) +
      theme_minimal()

    if (display_plots) {
      print(p)
    }
    if (!is.null(output_dir)) {
      ggsave(
        filename = file.path(output_dir, paste0(plot_key, ".png")),
        plot = p,
        width = 8,
        height = 5,
        dpi = 300
      )
    }
    if (pause_seconds > 0) {
      Sys.sleep(pause_seconds)
    }

    plot_results[[plot_key]] <- list(
      plot = p,
      observations = observation_df,
      predictions = prediction_df,
      metadata = list(
        replicate = replicate_id,
        noise_std = noise,
        sparsity_prob = sparsity,
        simulation_file = sim_path
      )
    )
  }

  plot_results
}

`%||%` <- function(lhs, rhs) {
  if (!is.null(lhs)) lhs else rhs
}

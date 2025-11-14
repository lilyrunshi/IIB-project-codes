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
  reticulate <- require_reticulate()

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

#' Compute the noise-free signal for a simulation replicate when available.
#'
#' @param sim_obj List produced by ``model3_simulate`` containing the exported
#'   arrays.
#' @param design_matrix Numeric matrix of covariates for the replicate.
#' @return Numeric vector representing the latent signal or ``NULL`` when it
#'   cannot be recovered.
compute_simulation_signal <- function(sim_obj, design_matrix) {
  if (!is.null(sim_obj$signal)) {
    signal <- as.numeric(sim_obj$signal)
    return(signal)
  }

  if (!is.null(sim_obj$w_true)) {
    w_true <- as.numeric(sim_obj$w_true)
    if (!is.null(design_matrix) && length(w_true) == ncol(design_matrix)) {
      return(as.numeric(design_matrix %*% w_true))
    }
  }

  latent <- sim_obj$latent %||% NULL
  if (!is.null(latent) && !is.null(latent$weights)) {
    weights <- as.numeric(latent$weights)
    if (!is.null(design_matrix) && length(weights) == ncol(design_matrix)) {
      return(as.numeric(design_matrix %*% weights))
    }
  }

  NULL
}

#' Extract the time axis used by the simulation.
#'
#' @param sim_obj Simulation output list.
#' @param n_obs Number of observations in the replicate.
#' @return Numeric vector of length ``n_obs`` describing the sampling locations.
extract_time_values <- function(sim_obj, n_obs, design_matrix = NULL, feature_index = 1L) {
  candidates <- list(sim_obj$time_points, sim_obj$time, sim_obj$times, sim_obj$t)

  dsc_time <- tryCatch(sim_obj$DSC_DEBUG$time, error = function(...) NULL)
  if (!is.null(dsc_time) && length(dsc_time) == n_obs) {
    candidates <- c(list(dsc_time), candidates)
  }

  for (candidate in candidates) {
    if (!is.null(candidate)) {
      candidate_vec <- as.numeric(candidate)
      if (length(candidate_vec) == n_obs) {
        return(candidate_vec)
      }
    }
  }

  if (!is.null(design_matrix) && is.matrix(design_matrix) && ncol(design_matrix) >= feature_index) {
    candidate <- design_matrix[, feature_index]
    candidate_vec <- as.numeric(candidate)
    if (length(candidate_vec) == n_obs) {
      return(candidate_vec)
    }
  }

  seq_len(n_obs)
}

#' Retrieve the fitted coefficient vector from an analysis module output.
#'
#' @param fit_obj The ``fit`` element loaded from a DSC analysis pickle file.
#' @param n_features Number of columns in the design matrix.
#' @return Numeric vector of length ``n_features`` or ``NULL`` if it cannot be
#'   extracted.
extract_weight_vector <- function(fit_obj, n_features) {
  if (is.null(fit_obj) || !length(fit_obj)) {
    return(NULL)
  }

  candidate_names <- c(
    "w_mean",
    "weights_mean",
    "weights_posterior_mean",
    "weights_map",
    "m_N",
    "coef_",
    "weights"
  )

  weights <- NULL
  for (name in candidate_names) {
    if (!is.null(fit_obj[[name]])) {
      candidate <- fit_obj[[name]]
      candidate <- if (is.list(candidate)) unlist(candidate, use.names = FALSE) else candidate
      candidate <- as.numeric(candidate)
      if (length(candidate) > 0L) {
        weights <- candidate
        break
      }
    }
  }

  if (is.null(weights)) {
    return(NULL)
  }

  if (!is.null(fit_obj$intercept_) && length(weights) < n_features) {
    weights <- c(weights, as.numeric(fit_obj$intercept_))
  }

  if (length(weights) > n_features) {
    weights <- weights[seq_len(n_features)]
  }

  if (length(weights) != n_features) {
    return(NULL)
  }

  weights
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
  feature_index <- as.integer(feature_index %||% 1L)
  if (is.na(feature_index) || feature_index < 1L) {
    feature_index <- 1L
  }

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

    design_matrix <- if (is.null(x)) NULL else as.matrix(x)
    if (is.null(design_matrix)) {
      warning("Simulation output at ", sim_path, " did not contain an 'x' matrix; skipping.")
      next
    }

    if (nrow(design_matrix) != length(y)) {
      warning(
        sprintf(
          "Design matrix row count (%d) did not match length of y (%d) in '%s'; skipping replicate.",
          nrow(design_matrix),
          length(y),
          basename(sim_path)
        )
      )
      next
    }

    replicate_id <- if (!is.null(meta$replicate)) meta$replicate else NA_integer_
    if (!is.null(replicates) && !is.na(replicate_id) && !(replicate_id %in% replicates)) {
      next
    }

    time_vals <- extract_time_values(sim_obj, length(y), design_matrix, feature_index)
    order_idx <- order(time_vals)

    signal_vec <- compute_simulation_signal(sim_obj, design_matrix)

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
      x_value = as.numeric(time_vals),
      y = as.numeric(y)
    )

    if (!is.null(signal_vec) && length(signal_vec) == nrow(design_matrix)) {
      observation_df$signal <- as.numeric(signal_vec)
      signal_df <- data.frame(
        sample = observation_df$sample,
        x_value = observation_df$x_value,
        signal = observation_df$signal
      )
      signal_df <- signal_df[order_idx, , drop = FALSE]
    } else {
      signal_df <- NULL
    }

    prediction_list <- list()
    weight_records <- list()
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
      fit_obj <- pred_obj$fit %||% NULL
      if (is.null(fit_obj)) {
        warning("Model output at ", pred_path, " did not contain a 'fit' object; skipping.")
        next
      }

      weights <- extract_weight_vector(fit_obj, ncol(design_matrix))
      if (is.null(weights)) {
        warning("Could not extract coefficient vector for ", pred_path, "; skipping.")
        next
      }

      module_prediction <- as.numeric(design_matrix %*% weights)
      if (length(module_prediction) != nrow(observation_df)) {
        warning(
          sprintf(
            "Derived prediction length mismatch for %s (got %d, expected %d); skipping.",
            basename(pred_path),
            length(module_prediction),
            nrow(observation_df)
          )
        )
        next
      }

      module_df <- data.frame(
        sample = observation_df$sample,
        x_value = observation_df$x_value,
        model = module,
        prediction = module_prediction
      )
      module_df <- module_df[order_idx, , drop = FALSE]
      prediction_list[[module]] <- module_df
      weight_records[[module]] <- weights
    }

    prediction_df <- if (length(prediction_list)) do.call(rbind, prediction_list) else NULL
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

    p <- ggplot() +
      geom_point(
        data = observation_df,
        aes(x = x_value, y = y),
        colour = "black",
        alpha = 0.6,
        size = 1.2
      )

    if (!is.null(signal_df)) {
      p <- p + geom_line(
        data = signal_df,
        aes(x = x_value, y = signal),
        colour = "black",
        linewidth = 1.0,
        alpha = 0.8
      )
    }

    p <- p + geom_line(
      data = prediction_df,
      aes(x = x_value, y = prediction, colour = model, group = model),
      linewidth = 0.7,
      alpha = 0.9
    ) +
      labs(
        title = plot_title,
        x = "Time",
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
      signal = signal_df,
      weights = weight_records,
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

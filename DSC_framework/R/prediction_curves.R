# Helper for plotting Model 3 simulation signals alongside model predictions.
#
# The DSC pipeline stores Python outputs as pickle files.  This script wraps the
# necessary plumbing to load the latent time points, observed responses, and
# regression weights so that each run can be visualised directly from R.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
})

require_dscquery <- function() {
  if (!requireNamespace("dscrutils", quietly = TRUE)) {
    stop(
      paste0(
        "Could not load dscrutils::dscquery(). Install the 'dscrutils' package",
        " or supply the DSC results manually via the `dsc_table` argument."),
      call. = FALSE
    )
  }
  tryCatch(
    getExportedValue("dscrutils", "dscquery"),
    error = function(e) {
      stop(
        paste0(
          "Could not load dscrutils::dscquery(). Install the 'dscrutils' package",
          " or supply the DSC results manually via the `dsc_table` argument.")
      ),
      call. = FALSE
    }
  )
}

require_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop(
      paste0(
        "Could not load reticulate. Install the 'reticulate' package so that ",
        "Python pickle artefacts emitted by DSC can be read from R."),
      call. = FALSE
    )
  }
  tryCatch(
    getExportedValue("reticulate", "py_run_string"),
    error = function(e) {
      stop(
        paste0(
          "Could not load reticulate. Install the 'reticulate' package so that ",
          "Python pickle artefacts emitted by DSC can be read from R."),
        call. = FALSE
      )
    }
  )
}

resolve_result_path <- function(path, root_dir) {
  if (is.null(path) || length(path) == 0 || is.na(path)) {
    return(NA_character_)
  }
  path <- path[[1]]
  if (!is.character(path)) {
    return(NA_character_)
  }
  if (path == "") {
    return(NA_character_)
  }
  is_absolute <- grepl("^/", path) || grepl("^[A-Za-z]:", path)
  if (!is_absolute && !is.null(root_dir)) {
    candidate <- file.path(root_dir, path)
  } else {
    candidate <- path
  }
  normalizePath(candidate, winslash = "/", mustWork = FALSE)
}

coerce_id_string <- function(column) {
  if (is.list(column)) {
    vapply(
      seq_along(column),
      function(i) {
        entry <- column[[i]]
        if (is.null(entry)) {
          return(sprintf("row%s", i))
        }
        if (is.character(entry) && length(entry) >= 1) {
          return(entry[1])
        }
        if (is.list(entry) && length(entry) >= 1 && is.character(entry[[1]])) {
          return(entry[[1]])
        }
        sprintf("row%s", i)
      },
      character(1)
    )
  } else if (is.character(column)) {
    column
  } else {
    as.character(column)
  }
}

load_pickle_object <- function(path) {
  py_run_string <- require_reticulate()
  if (!file.exists(path)) {
    stop(sprintf("Pickle file not found: %s", path), call. = FALSE)
  }
  reader <- py_run_string(sprintf(
    "import pickle\nwith open(r'''%s''', 'rb') as fh:\n    obj = pickle.load(fh)",
    path
  ))
  reticulate::py_to_r(reader$obj)
}

load_dsc_output <- function(value, root_dir) {
  if (is.null(value)) {
    return(NULL)
  }
  if (is.character(value) && length(value) == 1 && file.exists(value)) {
    ext <- tolower(tools::file_ext(value))
    if (ext == "rds") {
      return(readRDS(value))
    }
    if (ext %in% c("rda", "rdata")) {
      env <- new.env(parent = emptyenv())
      load(value, envir = env)
      objects <- as.list(env)
      if (length(objects) == 1) {
        return(objects[[1]])
      }
      return(objects)
    }
    if (ext %in% c("pkl", "pickle")) {
      return(load_pickle_object(value))
    }
    if (ext %in% c("npy", "npz")) {
      require_reticulate()
      np <- reticulate::import("numpy")
      arr <- np$load(value, allow_pickle = TRUE)
      return(reticulate::py_to_r(arr))
    }
  }
  if (is.character(value) && length(value) == 1 && !file.exists(value)) {
    resolved <- resolve_result_path(value, root_dir)
    if (!is.na(resolved) && file.exists(resolved)) {
      return(load_dsc_output(resolved, root_dir = NULL))
    }
  }
  if (is.list(value) && length(value) == 1 && is.character(value[[1]])) {
    resolved <- resolve_result_path(value[[1]], root_dir)
    if (!is.na(resolved) && file.exists(resolved)) {
      return(load_dsc_output(resolved, root_dir = NULL))
    }
  }
  value
}

compute_signal_from_weights <- function(time_points, weights, frequency = 1.0) {
  weights <- as.numeric(weights)
  if (length(weights) < 3) {
    stop(
      sprintf(
        "Expected at least three coefficients (sin, cos, intercept); got length %s",
        length(weights)
      ),
      call. = FALSE
    )
  }
  angular <- 2 * pi * frequency * time_points
  sin_component <- sin(angular) * weights[1]
  cos_component <- cos(angular) * weights[2]
  intercept_component <- rep(weights[3], length(time_points))
  sin_component + cos_component + intercept_component
}

extract_weights <- function(fit_object) {
  if (is.null(fit_object)) {
    return(NULL)
  }
  if (is.list(fit_object)) {
    if (!is.null(fit_object$w_mean)) {
      return(as.numeric(fit_object$w_mean))
    }
    if (!is.null(fit_object$m_N)) {
      return(as.numeric(fit_object$m_N))
    }
    if (!is.null(fit_object$weights)) {
      return(as.numeric(fit_object$weights))
    }
  }
  if (is.numeric(fit_object)) {
    return(as.numeric(fit_object))
  }
  stop("Could not find regression weights in fit object.", call. = FALSE)
}

read_model3_simulation <- function(row, root_dir, frequency = 1.0) {
  if (is.data.frame(row)) {
    row <- as.list(row)
  }
  latent <- load_dsc_output(row$simulate.latent, root_dir)
  time_points <- NULL
  if (is.list(latent) && !is.null(latent$time)) {
    time_points <- as.numeric(latent$time)
  }
  if (is.null(time_points)) {
    stop("Latent time points were not found in the simulation output.", call. = FALSE)
  }
  observed <- load_dsc_output(row$simulate.y, root_dir)
  if (is.list(observed) && length(observed) == 1) {
    observed <- observed[[1]]
  }
  observed <- as.numeric(observed)
  if (length(observed) != length(time_points)) {
    stop(
      sprintf(
        "Observed response length (%s) does not match time vector (%s).",
        length(observed),
        length(time_points)
      ),
      call. = FALSE
    )
  }

  weights <- load_dsc_output(row$simulate.w_true, root_dir)
  if (is.list(weights) && length(weights) == 1) {
    weights <- weights[[1]]
  }
  weights <- as.numeric(weights)
  if (length(weights) < 3) {
    stop("Simulation weights must include sin, cos, and intercept coefficients.", call. = FALSE)
  }

  tibble(
    time = time_points,
    observed = observed,
    true_signal = compute_signal_from_weights(time_points, weights, frequency = frequency)
  )
}

prepare_prediction_curves <- function(dsc_table, root_dir, frequency = 1.0) {
  if (!"simulate.latent" %in% names(dsc_table) ||
      !"simulate.y" %in% names(dsc_table) ||
      !"simulate.w_true" %in% names(dsc_table) ||
      !"analyze.fit" %in% names(dsc_table)) {
    stop(
      paste0(
        "The DSC table is missing required columns. Ensure that dscquery() was ",
        "invoked with targets including simulate.latent, simulate.y, simulate.w_true, and analyze.fit."
      ),
      call. = FALSE
    )
  }

  latent_keys <- coerce_id_string(dsc_table$simulate.latent)
  response_keys <- coerce_id_string(dsc_table$simulate.y)

  dataset_ids <- interaction(
    latent_keys,
    response_keys,
    drop = TRUE,
    lex.order = TRUE
  )

  dsc_table$dataset_id <- as.integer(dataset_ids)

  simulations <- dsc_table %>%
    select(dataset_id, simulate.latent, simulate.y, simulate.w_true, simulate.noise_std, simulate.sparsity_prob) %>%
    distinct() %>%
    rowwise() %>%
    mutate(data = list(read_model3_simulation(cur_data_all(), root_dir, frequency = frequency))) %>%
    ungroup()

  predictions <- dsc_table %>%
    mutate(
      fit_object = purrr::map(analyze.fit, load_dsc_output, root_dir = root_dir),
      weights = purrr::map(fit_object, extract_weights)
    ) %>%
    select(dataset_id, analyze, weights)

  list(simulations = simulations, predictions = predictions)
}

#' Plot Model 3 simulation curves with model predictions for each run.
#'
#' @param dsc_path Path to the DSC run directory containing `dsc-result.rds`.
#'   Ignored when `dsc_table` is supplied.
#' @param dsc_table Optional pre-loaded DSC table (e.g., the result of
#'   `dscrutils::dscquery(..., return.files = TRUE)`). When provided the DSC
#'   directory is not queried again.
#' @param output_dir Directory where PNG files should be written. Set to `NULL`
#'   to skip saving plots to disk.
#' @param display_plots Should plots be printed as they are generated?
#'   Defaults to `interactive()`.
#' @param pause_seconds Optional pause between plots.
#' @param frequency Sinusoidal frequency used in the design matrix. Defaults to
#'   1.0 to match the simulation module.
#' @return A named list of `ggplot` objects indexed by dataset identifier.
plot_model3_signal_predictions <- function(
    dsc_path = NULL,
    dsc_table = NULL,
    output_dir = file.path("plot_outputs", "model3_signal_predictions"),
    display_plots = interactive(),
    pause_seconds = 0,
    frequency = 1.0) {
  if (is.null(dsc_table)) {
    if (is.null(dsc_path)) {
      stop("Provide either `dsc_path` or a pre-loaded `dsc_table`.", call. = FALSE)
    }
    dsc_path <- normalizePath(dsc_path, winslash = "/", mustWork = TRUE)
    query_fun <- require_dscquery()
    message("Querying DSC outputs from: ", dsc_path)
    dsc_table <- query_fun(
      dsc.outdir = dsc_path,
      targets = c(
        "simulate.noise_std",
        "simulate.sparsity_prob",
        "simulate.latent",
        "simulate.y",
        "simulate.w_true",
        "analyze",
        "analyze.fit"
      ),
      return.files = TRUE
    )
  }

  if (!inherits(dsc_table, "tbl_df")) {
    dsc_table <- dplyr::as_tibble(dsc_table)
  }

  root_dir <- if (!is.null(dsc_path)) dsc_path else getwd()

  prep <- prepare_prediction_curves(dsc_table, root_dir = root_dir, frequency = frequency)

  simulations <- prep$simulations
  predictions <- prep$predictions

  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  plot_list <- vector("list", nrow(simulations))

  for (idx in seq_len(nrow(simulations))) {
    sim_row <- simulations[idx, ]
    dataset_id <- sim_row$dataset_id
    base_data <- sim_row$data[[1]]

    prediction_rows <- predictions %>%
      filter(dataset_id == !!dataset_id) %>%
      filter(purrr::map_lgl(weights, ~ !is.null(.x)))

    curve_data <- list()
    if (nrow(prediction_rows) > 0) {
      curve_data <- prediction_rows %>%
        mutate(
          curve = purrr::map2(
            weights,
            analyze,
            function(w, model_name) {
              tibble(
                time = base_data$time,
                value = compute_signal_from_weights(base_data$time, w, frequency = frequency),
                curve = paste0("Prediction: ", model_name)
              )
            }
          )
        ) %>%
        pull(curve)
    }

    tidy_curves <- dplyr::bind_rows(c(
      list(tibble(time = base_data$time, value = base_data$true_signal, curve = "True signal")),
      curve_data
    ))

    tidy_curves <- tidy_curves %>% arrange(time)

    plot_obj <- ggplot(base_data, aes(x = time, y = observed)) +
      geom_point(colour = "grey40", alpha = 0.7) +
      geom_line(
        data = tidy_curves,
        aes(x = time, y = value, colour = curve),
        linewidth = 1
      ) +
      labs(
        title = sprintf(
          "Model 3 signal vs predictions | noise=%s | sparsity=%s | run=%s",
          sim_row$simulate.noise_std,
          sim_row$simulate.sparsity_prob,
          dataset_id
        ),
        x = "Time",
        y = "Response",
        colour = "Curve"
      ) +
      theme_minimal()

    if (display_plots) {
      print(plot_obj)
    }

    if (!is.null(output_dir)) {
      ggsave(
        filename = file.path(
          output_dir,
          sprintf(
            "model3_signal_predictions_noise_%s_sparsity_%s_run_%s.png",
            sim_row$simulate.noise_std,
            sim_row$simulate.sparsity_prob,
            dataset_id
          )
        ),
        plot = plot_obj,
        width = 8,
        height = 5,
        dpi = 300
      )
    }

    if (pause_seconds > 0) {
      Sys.sleep(pause_seconds)
    }

    plot_list[[idx]] <- plot_obj
  }

  names(plot_list) <- sprintf("run_%s", simulations$dataset_id)
  plot_list
}

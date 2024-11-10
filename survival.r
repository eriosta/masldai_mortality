library(survival)
library(survminer)
library(tidyverse)

# Load the dataset
df <- read_csv("mortality_analysis.csv")

# Prepare the data
df$time <- df$permth_exm / 12  # Convert months to years
df <- df[!is.na(df$time) & !is.na(df$mortstat) & !is.na(df$Prediction_95_spec) &
           !is.na(df$Age_in_years_at_screening) & !is.na(df$HSSEX) & 
           !is.na(df$isfib4mod) & !is.na(df$isfib4high), ]

colnames(df) <- c("mortstat", "is_cardiac_mortality", "Prediction_95_spec", "Age", "Sex", 
                  "isfib4mod", "isfib4high", "permth_exm", "time")

# Create output folder
output_folder <- "Survival_Plots"
dir.create(output_folder, showWarnings = FALSE)

# Function to rename variables for plotting
rename_variable <- function(var, value) {
  if (var == "Prediction_95_spec") {
    return(ifelse(value == 1, "≥0.4269", "<0.4269"))
  } else if (var == "isfib4high") {
    return(ifelse(value == 1, "≥2.67", "<2.67"))
  } else if (var == "isfib4mod") {
    return(ifelse(value == 1, "≥1.30", "<1.30"))
  } else {
    return(as.character(value))
  }
}

rename_outcome <- function(outcome) {
  if (outcome == "mortstat") {
    return("All-cause mortality")
  } else if (outcome == "is_cardiac_mortality") {
    return("Cardiovascular-related mortality")
  } else {
    return(outcome)
  }
}

# Function to generate plots
generate_plot <- function(variable, outcome, adjusted, risk_table) {
  if (adjusted) {
    # Adjusted Cox proportional hazards model
    cox_model <- coxph(as.formula(paste0("Surv(time, ", outcome, ") ~ ", variable, " + Age + Sex")), data = df)
    
    # Create newdata for adjusted survival curves
    newdata <- data.frame(
      Age = mean(df$Age, na.rm = TRUE),
      Sex = mean(df$Sex, na.rm = TRUE)
    )
    # Replicate rows to match the number of groups (0 and 1)
    newdata <- newdata[rep(1, 2), ]
    # Add the variable column with values 0 and 1
    newdata[[variable]] <- c(0, 1)
    
    surv_fit <- survfit(cox_model, newdata = newdata)
    risk_table_flag <- FALSE
    plot_width <- 6
    plot_height <- 6
  } else {
    # Unadjusted Kaplan-Meier
    surv_fit <- survfit(as.formula(paste0("Surv(time, ", outcome, ") ~ ", variable)), data = df)
    risk_table_flag <- risk_table
    plot_width <- 5
    plot_height <- 6
  }
  
  # Calculate Hazard Ratios and Confidence Intervals (only for adjusted)
  if (adjusted) {
    cox_summary <- summary(cox_model)
    hr <- round(cox_summary$coefficients[variable, "exp(coef)"], 2)
    hr_lower <- round(cox_summary$conf.int[variable, "lower .95"], 2)
    hr_upper <- round(cox_summary$conf.int[variable, "upper .95"], 2)
    hr_text <- paste0("HR: ", hr, " (95% CI: ", hr_lower, "-", hr_upper, ")")
  } else {
    hr_text <- ""
  }
  
  # Define theme
  my_theme <- theme(
    panel.background = element_blank(),
    panel.border = element_rect(fill = NA, color = "black", size = 0.8),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.position = "top",
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12)
  )
  
  # Rename variables and outcome for plot
  legend_title <- switch(variable,
                         "Prediction_95_spec" = "FibroX",
                         "isfib4mod" = "FIB-4",
                         "isfib4high" = "FIB-4")
  outcome_title <- rename_outcome(outcome)
  legend_labels <- c(rename_variable(variable, 0), rename_variable(variable, 1))
  
  # Define plot title
  adj_status <- ifelse(adjusted, "Adjusted", "Unadjusted")
  title <- paste(adj_status, legend_title, "-", outcome_title)
  
  # Plot
  plot <- ggsurvplot(
    surv_fit,
    data = df,
    conf.int = TRUE,
    legend.labs = legend_labels,
    legend.title = legend_title,
    palette = c("#0072B2", "#D55E00"),
    xlab = "Years",
    ylab = "Survival Probability",
    title = title,
    ggtheme = my_theme,
    risk.table = risk_table_flag,
    risk.table.title = "Number at Risk",
    size = 1
  )
  
  # Add HR annotation (only for adjusted)
  if (adjusted) {
    plot$plot <- plot$plot +
      annotate(
        "text",
        x = 1, y = 0.1,  # Adjust x and y coordinates for placement
        label = hr_text,
        size = 5,
        hjust = 0
      )
  }
  
  # Save the plot
  file_name <- paste0(output_folder, "/", adj_status, "_", variable, "_vs_", outcome, ".tiff")
  tiff(file_name, width = plot_width, height = plot_height, units = "in", res = 300)
  print(plot)
  dev.off()
}

# Generate plots
variables <- c("Prediction_95_spec", "isfib4mod", "isfib4high")
outcomes <- c("mortstat", "is_cardiac_mortality")
for (variable in variables) {
  for (outcome in outcomes) {
    # Unadjusted with risk tables
    generate_plot(variable, outcome, adjusted = FALSE, risk_table = TRUE)
    # Adjusted without risk tables
    generate_plot(variable, outcome, adjusted = TRUE, risk_table = FALSE)
  }
}

message("Plots have been generated and saved in the 'Survival_Plots' folder.")

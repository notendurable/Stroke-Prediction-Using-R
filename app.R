# Load required libraries
library(shiny)
library(tidyverse)
library(ggplot2)

# Load pre-saved data
evaluation_results <- read.csv("evaluation_metrics.csv")  # Evaluation metrics table
confusion_matrices <- readRDS("confusion_matrices.rds")   # Confusion matrices

# Initialize Models with Error Handling
models <- list(
  Logistic_Regression = tryCatch(readRDS("logistic_regression_model.rds"), error = function(e) NULL),
  Random_Forest = tryCatch(readRDS("random_forest_model.rds"), error = function(e) NULL),
  Support_Vector_Machine = tryCatch(readRDS("support_vector_machine_model.rds"), error = function(e) NULL),
  Decision_Tree = tryCatch(readRDS("decision_tree_model.rds"), error = function(e) NULL),
  XGBoost = tryCatch(readRDS("xgboost_model.rds"), error = function(e) NULL)
)

# Define UI
ui <- fluidPage(
  titlePanel("Stroke Risk Dashboard"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("age", "Age (Years):", min = 0, max = 100, value = 50),
      numericInput("glucose", "Average Glucose Level (mg/dL):", value = 101, min = 0),
      numericInput("bmi", "BMI:", value = 25, min = 0),
      selectInput("gender", "Gender:", choices = c("Female", "Male")),
      selectInput("hypertension", "Hypertension:", choices = c("No", "Yes")),
      selectInput("heart_disease", "Heart Disease:", choices = c("No", "Yes")),
      selectInput("ever_married", "Ever Married:", choices = c("No", "Yes")),
      selectInput("work_type", "Work Type:", choices = c("children", "Govt_job", "Never_worked", "Private", "Self-employed")),
      selectInput("residence", "Residence Type:", choices = c("Rural", "Urban")),
      selectInput("smoking_status", "Smoking Status:", choices = c("never smoked", "formerly smoked", "smokes")),
      actionButton("predict", "Predict Stroke Risk"),
      downloadButton("downloadEval", "Download Evaluation Metrics"),
      downloadButton("downloadPred", "Download Predictions")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Home Page",
                 h3("Stroke Risk Prediction Project Overview", style = "font-weight: bold"),
                 p("Stroke is the second leading cause of death globally, accounting for approximately 11% of total deaths (WHO). Early identification of stroke risk is critical for timely intervention and improved patient outcomes. This project leverages machine learning to predict stroke risk using demographic, clinical, and lifestyle data."),
                 h4("Objectives", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Develop and evaluate machine learning models to predict stroke risk in R and deploy the models."),
                   tags$li("Identify key predictors contributing to stroke."),
                   tags$li("Provide actionable insights to guide clinical and public health strategies.")
                 ),
                 h4("Data Summary", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Demographic Factors: Age, gender, marital status, residence type."),
                   tags$li("Clinical Factors: Hypertension, heart disease, average glucose level, BMI."),
                   tags$li("Lifestyle Factors: Smoking status, work type.")
                 ),
                 p("The target variable is stroke (binary: 'Yes' or 'No'). Due to class imbalance (fewer stroke-positive cases), model evaluation emphasized metrics beyond accuracy."),
                 h4("Models Used", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Logistic Regression: Established baseline performance; interpretable but limited in ranking probabilities."),
                   tags$li("Random Forest: High sensitivity, robust for feature importance but struggled with probability ranking."),
                   tags$li("Support Vector Machine (SVM): Balanced precision and recall, with improved class separation."),
                   tags$li("Decision Tree: Simple and interpretable; good for binary classification."),
                   tags$li("XGBoost: The top-performing model; handled complex relationships and offered strong predictive performance.")
                 ),
                 h4("Key Findings", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Accuracy ranged from 91.86% (SVM) to 94.75% (Logistic Regression, Random Forest, Decision Tree)."),
                   tags$li("XGBoost achieved the highest ROC AUC (0.16) among models, balancing sensitivity (99.75%) and precision (94.75%).")
                 ),
                 h4("Recommendations", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Deploy XGBoost or Random Forest models in clinical settings for robust stroke prediction."),
                   tags$li("Enhance dataset quality by addressing class imbalance through techniques like SMOTE or collecting more balanced data."),
                   tags$li("Educate stakeholders on leveraging model outputs for proactive care."),
                   tags$li("Validate models with external datasets to ensure generalizability.")
                 ),
                 h4("Future Directions", style = "font-weight: bold"),
                 tags$ul(
                   tags$li("Combine model predictions for improved performance through ensemble methods."),
                   tags$li("Integrate predictive tools into decision-support systems for scalable deployment."),
                   tags$li("Expand dataset collection to include diverse populations, improving model robustness and inclusivity.")
                 )
        ),
        tabPanel("Model Predictions",
                 h4("Key Features"),
                 p("1. ", span("Color-Coded Probabilities:", style = "font-weight: bold"),
                   " Predictions are displayed with color coding for low, moderate, and high-risk probabilities."),
                 p("    - ", span("Low Risk:", style = "font-weight: bold; color: green"), " (Green, less than 20%)."),
                 p("    - ", span("Moderate Risk:", style = "font-weight: bold; color: yellow"), " (Yellow, between 20% and 50%)."),
                 p("    - ", span("High Risk:", style = "font-weight: bold; color: red"), " (Red, above 50%)."),
                 hr(),
                 h4("Predicted Probability of Stroke"),
                 uiOutput("predictions_output")
        ),
        tabPanel("Model Performance Metrics",
                 h4("Explanation of Model Metrics", style = "font-weight: bold"),
                 div(
                   p("Metrics Overview:"),
                   tags$ul(
                     tags$li(tags$span("Accuracy: ", style = "font-weight: bold"), "Measures the overall correctness of predictions."),
                     tags$li(tags$span("Kappa (kap): ", style = "font-weight: bold"), "Assesses agreement between predictions and actual values, accounting for chance."),
                     tags$li(tags$span("ROC AUC: ", style = "font-weight: bold"), "Evaluates the model's ability to separate positive and negative classes."),
                     tags$li(tags$span("Precision: ", style = "font-weight: bold"), "Proportion of positive predictions that are correct."),
                     tags$li(tags$span("Recall: ", style = "font-weight: bold"), "Proportion of actual positives correctly identified."),
                     tags$li(tags$span("F-measure (f_meas): ", style = "font-weight: bold"), "A balanced metric that combines precision and recall.")
                   ),
                   hr(),
                   h4("Model Performance Analysis"),
                   p("Model performance insights are as follows:"),
                   tags$ul(
                     tags$li(tags$span("Logistic Regression: ", style = "font-weight: bold"), "High accuracy but struggles with ROC AUC."),
                     tags$li(tags$span("Random Forest: ", style = "font-weight: bold"), "Similar to Logistic Regression but with higher sensitivity."),
                     tags$li(tags$span("Support Vector Machine: ", style = "font-weight: bold"), "Better class separation with improved ROC AUC."),
                     tags$li(tags$span("Decision Tree: ", style = "font-weight: bold"), "Simple and interpretable, suitable for binary classification."),
                     tags$li(tags$span("XGBoost: ", style = "font-weight: bold"), "Strong overall performance but requires careful tuning.")
                   )
                 ),
                 selectInput("modelPerformanceSelect", "Select a Model to View Metrics:", 
                             choices = unique(evaluation_results$Model)),
                 tableOutput("selectedModelMetrics")
        ),
        tabPanel("Model Confusion Matrix",
                 h4("Understanding Confusion Matrix", style = "font-weight: bold"),
                 div(
                   p("Each row in the table represents a specific combination of predicted and actual classes:"),
                   tags$ul(
                     tags$li(tags$span("Prediction = No, Truth = No: ", style = "font-weight: bold"), "True Negatives."),
                     tags$li(tags$span("Prediction = Yes, Truth = No: ", style = "font-weight: bold"), "False Positives."),
                     tags$li(tags$span("Prediction = No, Truth = Yes: ", style = "font-weight: bold"), "False Negatives."),
                     tags$li(tags$span("Prediction = Yes, Truth = Yes: ", style = "font-weight: bold"), "True Positives.")
                   ),
                   hr(),
                   h4("Key Observations and Performance Implications"),
                   tags$ul(
                     tags$li(tags$span("Logistic Regression: ", style = "font-weight: bold"), "Identifies at least one true positive but struggles with precision."),
                     tags$li(tags$span("Random Forest: ", style = "font-weight: bold"), "Misses true positives, indicating poor recall."),
                     tags$li(tags$span("Support Vector Machine: ", style = "font-weight: bold"), "Balanced precision and recall but high false positives."),
                     tags$li(tags$span("XGBoost: ", style = "font-weight: bold"), "Strong recall but struggles with class imbalance.")
                   )
                 ),
                 selectInput("modelSelect", "Select a Model to View Confusion Matrix:", 
                             choices = names(confusion_matrices)),
                 tableOutput("confMatrix")
        )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  # Helper function to prepare user input data
  prepare_data <- reactive({
    tibble(
      age = input$age,
      avg_glucose_level = input$glucose,
      bmi = input$bmi,
      gender = input$gender,
      hypertension = input$hypertension,
      heart_disease = input$heart_disease,
      ever_married = input$ever_married,
      work_type = input$work_type,
      Residence_type = input$residence,
      smoking_status = input$smoking_status
    )
  })
  
  # Render Predictions
  observeEvent(input$predict, {
    new_data <- prepare_data()
    output$predictions_output <- renderUI({
      predictions <- lapply(names(models), function(model_name) {
        model <- models[[model_name]]
        if (is.null(model)) {
          return(HTML(paste0("<p style='color:red;'>", model_name, ": Model not available.</p>")))
        }
        pred_prob <- predict(model, new_data, type = "prob")[[".pred_Yes"]]
        prob <- round(pred_prob * 100, 2)
        color <- if (prob < 20) "green" else if (prob < 50) "yellow" else "red"
        HTML(paste0(
          "<p style='color:", color, ";'>", model_name, 
          ": Predicted Probability of Stroke = ", prob, "%</p>"
        ))
      })
      do.call(tagList, predictions)
    })
  })
  
  # Render Model Metrics
  output$selectedModelMetrics <- renderTable({
    req(input$modelPerformanceSelect)
    evaluation_results %>% filter(Model == input$modelPerformanceSelect)
  })
  
  # Render Confusion Matrix
  output$confMatrix <- renderTable({
    req(input$modelSelect)
    confusion_matrices[[input$modelSelect]]$table
  })
  
  # Download Evaluation Metrics
  output$downloadEval <- downloadHandler(
    filename = function() { "evaluation_metrics.csv" },
    content = function(file) {
      write.csv(evaluation_results, file, row.names = FALSE)
    }
  )
  
  # Download Predictions
  predictions_data <- reactive({
    req(input$predict)
    new_data <- prepare_data()
    pred_results <- lapply(names(models), function(model_name) {
      model <- models[[model_name]]
      if (!is.null(model)) {
        pred_prob <- predict(model, new_data, type = "prob")[[".pred_Yes"]]
        return(data.frame(Model = model_name, Probability = round(pred_prob * 100, 2)))
      }
      return(data.frame(Model = model_name, Probability = NA))
    })
    do.call(rbind, pred_results)
  })
  
  output$downloadPred <- downloadHandler(
    filename = function() { "predictions.csv" },
    content = function(file) {
      write.csv(predictions_data(), file, row.names = FALSE)
    }
  )
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
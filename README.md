Real-Time Context-Aware Text Classification & Live Stream Routing
📌 Project Overview
This project builds an automated customer feedback classification and routing system for unstructured, high-velocity customer messages.
Instead of using only a simple `category → department` lookup, the system combines:
TF-IDF text representation
Unigram + bigram features
Machine-learning classification
Model confidence scoring
Urgency-language detection
Context-aware priority escalation
Manual triage for low-confidence predictions
Real-time stream simulation
Live routing dashboard
Saved trained model pipeline
The complete workflow is implemented in `Text_classification.ipynb`.
---
🎯 Project Goal
The goal is to automatically process incoming customer feedback and determine:
What category does the message belong to?
How confident is the model?
Which department should handle it?
What priority should it receive?
Should it be automatically routed or sent for manual review?
The supported feedback categories are:
`Billing`
`Delivery`
`Product`
`Technical Support`
`General Inquiry`
---
✨ Key Features
🤖 Text Classification
Uses TF-IDF with unigrams and bigrams to classify customer feedback.
🧠 Context-Aware Routing
The system considers more than the predicted category. It also evaluates:
Base category priority
Urgency/frustration language
Model confidence
🚨 Priority Escalation
Urgent language can increase the priority of a message.
For example:
```text
"My package is STILL NOT here, this is unacceptable!"
```
can receive a higher priority than an ordinary delivery question.
🛡️ Confidence-Based Manual Review
If the model's highest predicted probability is below:
```text
0.40
```
the message is not automatically assigned to a department. Instead, it is routed to:
```text
Triage / Manual Review Queue
```
with:
```text
Priority: Needs Review
```
⚡ Real-Time Stream Simulation
The notebook simulates a live stream of incoming feedback and processes messages one by one.
📊 Live Dashboard
The project visualizes:
Messages routed to each team
Messages by priority level
💾 Model Persistence
The trained TF-IDF + classifier pipeline is saved as:
```text
feedback_router_model.joblib
```
---
📂 Dataset
The notebook expects a dataset named:
```text
customer_feedback.csv
```
The dataset contains:
1,000 customer feedback messages
5 categories
200 messages per category
No missing values
Categories
Category	Number of Messages
Billing	200
Delivery	200
Product	200
Technical Support	200
General Inquiry	200
Total	1,000
The dataset is perfectly balanced, so accuracy is considered a fair evaluation metric in the notebook.
The notebook also notes that duplicate rows are present because the feedback is templated. These duplicates are expected for this synthetic/labelled support dataset and are not removed.
---
🏗️ System Architecture
```text
                 Customer Feedback
                         │
                         ▼
                  Text Cleaning
                         │
                         ▼
                TF-IDF Vectorization
                (Unigrams + Bigrams)
                         │
                         ▼
                ML Classification
                         │
                         ▼
              Predicted Category
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
          Confidence Check   Priority Layer
                  │             │
          Low Confidence    Base Priority
                  │             │
                  ▼        Urgency Detection
          Manual Triage          │
                  │              ▼
                  │       Priority Escalation
                  │              │
                  └──────┬───────┘
                         ▼
                  Department Routing
                         │
                         ▼
                  Live Dashboard
```
---
🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
WordCloud
Joblib
Machine Learning
TF-IDF Vectorization
Multinomial Naive Bayes
Logistic Regression
Linear SVM
Calibrated Classifier
Cross-validation
Development Environment
Jupyter Notebook
---
📁 Project Structure
```text
Real-Time-Text-Classification/
│
├── Text_classification.ipynb
├── customer_feedback.csv
├── feedback_router_model.joblib
└── README.md
```
---
🔄 Project Workflow
1. Import Libraries
The notebook imports the required tools for:
Data processing
Visualization
Text processing
Machine learning
Model evaluation
Model persistence
A fixed random state of `42` is used for reproducibility.
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```
---
2. Load & Explore the Dataset
The dataset is loaded using Pandas:
```python
df = pd.read_csv("customer_feedback.csv")
```
The notebook performs basic checks for:
Dataset shape
Missing values
Duplicate rows
Category distribution
```python
df.isnull().sum()
df.duplicated().sum()
df["Category"].value_counts()
```
---
3. Exploratory Data Analysis
The project visualizes:
Feedback Volume
A category count plot shows the number of messages belonging to each category.
Message Length
The notebook calculates the number of words in each feedback message and displays its distribution.
Word Clouds
Separate word clouds are generated for each category to show common words and patterns.
---
4. Light Text Preprocessing
The dataset already contains clean, short, natural-language sentences.
Therefore, the notebook intentionally avoids heavy preprocessing such as aggressive stemming or stopword removal.
This is important because words such as:
```text
not
```
and numbers such as order IDs can contain useful classification information.
The implemented cleaning function:
```python
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```
The preprocessing performs:
Lowercasing
Whitespace normalization
Removal of stray punctuation/noise
Preservation of letters, numbers, and `#`
---
5. Train/Test Split
The cleaned feedback is used as the input feature and `Category` as the target.
```python
X = df["clean_feedback"]
y = df["Category"]
```
The data is split into:
80% training data
20% testing data
The split is stratified so that the category distribution remains balanced.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)
```
---
6. TF-IDF Feature Engineering
The project uses TF-IDF instead of a simple word-count representation.
The vectorizer uses:
```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)
```
Why TF-IDF?
TF-IDF reduces the influence of words that appear frequently across many messages and emphasizes words that are more distinctive to a category.
Why Unigrams + Bigrams?
The model can learn both:
```text
single words
```
and phrases such as:
```text
not working
payment failed
```
This allows the representation to capture short contextual phrases rather than relying only on isolated words.
---
🤖 7. Model Training & Comparison
Three candidate classifiers are trained using the same TF-IDF representation.
Model 1 — Multinomial Naive Bayes
```python
MultinomialNB()
```
A fast and strong baseline for text classification.
Model 2 — Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```
Model 3 — Calibrated Linear SVM
```python
CalibratedClassifierCV(
    LinearSVC(random_state=42),
    cv=3
)
```
The Linear SVM is calibrated so that the system can obtain confidence/probability estimates required by the routing logic.
---
📏 Model Evaluation
The notebook compares models using:
Test Accuracy
Macro F1 Score
5-Fold Cross-Validation Accuracy
```python
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
cv_acc = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5
).mean()
```
The model with the highest Macro F1 is selected as the final model:
```python
best_model_name = results_df.loc[0, "Model"]
model = fitted_pipelines[best_model_name]
```
The notebook describes the final classifier as achieving very high accuracy on this dataset, because the feedback messages are short and category-distinctive.
> The exact numerical accuracy should be taken from the output generated when the notebook is executed; this README does not invent a score that is not explicitly available from the project source.
---
📊 8. Evaluation of the Best Model
The selected model is evaluated on the held-out test set.
The notebook generates:
Accuracy
```python
accuracy_score(y_test, y_pred)
```
Classification Report
The report provides:
Precision
Recall
F1-score
Support
Confusion Matrix
A confusion matrix is generated to visualize correct and incorrect predictions across the five categories.
---
🚦 9. Context-Aware Priority Layer
A major feature of this project is that classification is not the final routing decision.
The system adds a context-aware priority layer.
Department Mapping
Category	Department
Billing	Finance Team
Delivery	Logistics Team
Product	Quality Assurance Team
Technical Support	IT Support Team
General Inquiry	Customer Care Team
---
Base Priority
Category	Base Priority
Technical Support	High
Billing	High
Delivery	Medium
Product	Medium
General Inquiry	Low
---
Urgency Detection
The system checks for urgency/frustration indicators such as:
```text
urgent
immediately
asap
right now
still not
again
unacceptable
furious
angry
worst
terrible
never received
refund now
escalate
charged twice
```
If urgency language is detected, the priority is increased by one level.
The maximum level is:
```text
Critical
```
---
Confidence Threshold
The model uses:
```python
CONFIDENCE_THRESHOLD = 0.40
```
If the highest predicted probability is below `0.40`, the message is considered uncertain.
It is routed as:
```text
Category: Uncertain
Department: Triage / Manual Review Queue
Priority: Needs Review
```
This prevents low-confidence predictions from being automatically assigned to the wrong team.
---
🧩 10. Unified Routing Function
The central function is:
```python
route_message(raw_text)
```
It performs the complete decision process:
```text
Raw Message
     ↓
Clean Text
     ↓
Predict Category
     ↓
Calculate Confidence
     ↓
Confidence < 0.40?
     │
   Yes ──────────► Manual Review
     │
    No
     ↓
Assign Department
     ↓
Assign Base Priority
     ↓
Check Urgency Language
     ↓
Escalate if Required
     ↓
Return Routing Result
```
Example output structure:
```python
{
    "message": "...",
    "category": "Billing",
    "confidence": 0.95,
    "department": "Finance Team",
    "priority": "Critical"
}
```
The exact confidence value depends on the trained model and input message.
---
⚡ 11. Real-Time Stream Simulation
The notebook simulates a high-velocity incoming feedback stream.
The simulation combines:
12 randomly selected held-out test messages
3 manually written urgent messages
The messages are shuffled and processed one at a time.
A small delay is introduced:
```python
time.sleep(0.05)
```
This imitates the arrival of messages from a live message queue.
The notebook describes this architecture as similar in pattern to consumers reading events from systems such as:
Kafka
RabbitMQ
Kinesis
The simulation processes each message through:
```text
Message Arrival
      ↓
Classification
      ↓
Confidence
      ↓
Priority
      ↓
Department
      ↓
Running Counters
```
---
📊 12. Live Routing Dashboard
After processing the simulated stream, the notebook generates a dashboard containing two visualizations.
Messages Routed per Team
Shows how many messages were routed to each department.
Messages by Priority Level
Shows the distribution across:
```text
Critical
High
Medium
Low
Needs Review
```
This provides a quick snapshot of the stream's routing behavior.
---
🧪 13. Interactive Single-Message Test
The notebook allows a support agent to enter a message manually:
```python
feedback = input("Enter customer feedback: ")
result = route_message(feedback)
```
The system returns:
```text
Category
Confidence
Priority
Route To
```
This provides a simple way to test the routing model interactively.
---
💾 14. Saved Machine Learning Pipeline
The fitted pipeline is saved using Joblib:
```python
joblib.dump(model, "feedback_router_model.joblib")
```
The generated model file is:
```text
feedback_router_model.joblib
```
It contains the fitted pipeline used by the notebook, allowing the model to be loaded without retraining.
Reload the model
```python
import joblib

model = joblib.load("feedback_router_model.joblib")
```
> Note: The saved `.joblib` file contains the trained classification pipeline. The notebook's `route_message()` function additionally contains the project's priority, urgency, department mapping, and confidence-routing logic.
---
▶️ How to Run
1. Install Python Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud joblib jupyter
```
2. Place the Files Together
Make sure these files are available:
```text
Text_classification.ipynb
customer_feedback.csv
feedback_router_model.joblib
```
For a fresh training run, the notebook expects:
```text
customer_feedback.csv
```
in the working directory.
3. Start Jupyter Notebook
```bash
jupyter notebook
```
4. Open the Notebook
Open:
```text
Text_classification.ipynb
```
5. Run the Notebook
Run the cells from top to bottom.
The notebook will:
Load the dataset
Explore the data
Clean feedback text
Create TF-IDF features
Train three classifiers
Compare their performance
Select the best model
Evaluate the model
Apply priority and urgency logic
Simulate a live stream
Generate the routing dashboard
Allow interactive testing
Save the trained pipeline
---
📌 Example Routing Logic
Example 1 — Billing
```text
The payment failed and I was charged twice.
```
Expected behavior:
```text
Category  → Billing
Department → Finance Team
Priority   → Elevated because "charged twice" is an urgency marker
```
Example 2 — General Inquiry
```text
Hi, just wondering what your return policy is.
```
Expected behavior:
```text
Category → General Inquiry
Department → Customer Care Team
Priority → Low
```
Example 3 — Urgent Delivery Issue
```text
My package is STILL NOT here, this is unacceptable,
I need this escalated immediately!
```
Expected behavior:
```text
Category → Delivery
Department → Logistics Team
Priority → Escalated toward Critical
```
The exact predicted category/confidence is determined by the trained model.
---
📈 Project Strengths
Uses a complete NLP classification pipeline.
Compares multiple machine-learning algorithms.
Uses TF-IDF with bigrams for short contextual phrases.
Uses stratified train/test splitting.
Includes 5-fold cross-validation.
Evaluates the best model with a classification report and confusion matrix.
Adds model confidence to routing decisions.
Uses urgency language for priority escalation.
Provides a manual-review fallback.
Simulates real-time message processing.
Includes a live routing dashboard.
Saves the trained model for reuse.
---
⚠️ Current Limitations
The current notebook is a simulation rather than a production deployment.
Important limitations include:
The dataset is synthetic/labelled support data.
The dataset contains templated duplicate messages.
The model relies on TF-IDF rather than semantic embeddings.
Urgency detection is based on predefined regular-expression patterns.
The stream is simulated inside a Jupyter Notebook.
No actual Kafka, RabbitMQ, or Kinesis consumer is connected.
The saved Joblib file contains the classification pipeline, while the routing logic remains in notebook code.
The confidence threshold of `0.40` is configured in the notebook and is not presented as a statistically optimized production threshold.
---
🚀 Future Improvements
The notebook suggests several directions for a production-ready system.
1. Sentence Embeddings
Replace TF-IDF with sentence embeddings such as:
```text
sentence-transformers
```
This could better handle more varied and informal customer feedback.
2. Human Feedback Loop
Messages sent to:
```text
Triage / Manual Review Queue
```
could be manually labelled and periodically added to the training dataset.
This would allow the model to improve over time.
3. Real Message Broker
Replace the notebook simulation with an actual message consumer connected to:
```text
Kafka
RabbitMQ
Kinesis
```
4. Production API
Expose `route_message()` through an API so that other applications can submit customer feedback programmatically.
5. Monitoring
A production deployment could monitor:
Model confidence
Category distribution
Manual-review rate
Priority distribution
Routing errors
Data drift
Model performance over time
---
🧠 End-to-End Concept
The core idea of this project can be summarized as:
```text
Customer Feedback
        ↓
   Text Cleaning
        ↓
    TF-IDF NLP
        ↓
 ML Classification
        ↓
 Predicted Category
        ↓
 Confidence Check
        ↓
 ┌───────────────┐
 │               │
Low Confidence  Confident
 │               │
 ▼               ▼
Manual Review   Department
                Routing
                    ↓
              Base Priority
                    ↓
             Urgency Detection
                    ↓
             Priority Escalation
                    ↓
             Real-Time Dashboard
```
---
📜 Conclusion
This project demonstrates an end-to-end real-time, context-aware customer feedback routing workflow.
The system does more than classify text. It combines:
NLP classification + confidence scoring + urgency detection + priority escalation + department routing + manual triage.
The result is a routing architecture where each incoming message can be processed individually:
```text
Message arrives
      ↓
Classify
      ↓
Check confidence
      ↓
Determine priority
      ↓
Route to the appropriate team
      ↓
Send uncertain messages to manual review
```
This makes the project a practical demonstration of how traditional text classification can be extended into a more context-aware automated support-routing workflow.
---
👨‍💻 Author
Dheen Seenivasan
BCA Student | Aspiring IT Professional
---
⭐ Project Highlights
```text
✔ NLP Text Classification
✔ TF-IDF Unigrams + Bigrams
✔ 5 Customer Feedback Categories
✔ 3 ML Models Compared
✔ Cross-Validation
✔ Confidence-Aware Routing
✔ Urgency Detection
✔ Priority Escalation
✔ Manual Triage Fallback
✔ Real-Time Stream Simulation
✔ Live Routing Dashboard
✔ Interactive Message Testing
✔ Joblib Model Persistence
```
---
📄 License
This project is intended for educational, learning, and portfolio purposes.

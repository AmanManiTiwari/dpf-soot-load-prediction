# Predictive Maintenance Pipeline: DPF Soot Load Prediction System

**Author:** Aman Mani Tiwari  
**Role Applied:** Data Science Intern – Tensor Planet  
**Tools Used:** Python, NumPy, Pandas, Scikit-learn, FastAPI  

---

## 1. Overview & Objective

Commercial diesel vehicles equipped with Diesel Particulate Filters (DPF) continuously accumulate soot during normal operation. Excessive soot loading can result in engine derate events, forced regenerations, increased fuel consumption, and unplanned downtime.

The objective of this assignment is to design an **end-to-end predictive maintenance pipeline** that estimates DPF soot load using vehicle telemetry data and provides **proactive regeneration recommendations**, while balancing false alarms and operational cost.

This solution focuses on:

- Structured synthetic data generation  
- Feature engineering informed by engineering intuition  
- Thoughtful problem framing and modeling tradeoffs  
- Production and robustness considerations  

---

## 2. Data Generation Approach

### 2.1 Assumptions

Since real-world DPF telemetry data is proprietary, synthetic datasets were generated with the following assumptions:

- Soot accumulation increases gradually based on engine load and operating conditions  
- Differential pressure increases as soot accumulates  
- Higher exhaust temperatures improve regeneration effectiveness  
- Sensor readings contain noise and mild drift, similar to real vehicle data  

The goal was **internal consistency**, not precise physical modeling.

---

### 2.2 Datasets Generated

#### 1. Sensor Telemetry Dataset

- **Granularity:** Every 5 minutes  
- **Duration:** 30 days  
- **Vehicles:** 20  

**Fields include:**
- Engine load  
- Exhaust temperature (pre & post DPF)  
- Differential pressure  
- Exhaust flow rate  
- Vehicle speed and RPM  
- Ambient temperature  
- Ground-truth soot load (used only for training)  

---

#### 2. Maintenance / Regeneration Records

Event-based dataset containing:

- Vehicle ID  
- Regeneration timestamp  
- Regeneration type (active regeneration)  

Events are triggered when soot load exceeds a defined threshold, simulating real maintenance logs.

---

#### 3. Trip Characteristics (Conceptual)

Trip-level aggregates such as trip duration, distance, and driving pattern were conceptually modeled and can be integrated in future iterations to enrich driving behavior features.

---

## 3. Data Engineering & Feature Design

### 3.1 Feature Engineering

Key engineered features include:

- **Rolling averages of exhaust temperature**  
  Capture sustained thermal conditions indicating regeneration opportunity.

- **Rolling averages of differential pressure**  
  Smooth noisy sensor readings and represent soot buildup trends.

- **Temperature delta (pre vs post DPF)**  
  Proxy for regeneration effectiveness.

All features are vehicle-aware and time-ordered to prevent data leakage.

---

### 3.2 Data Quality Checks

Basic validation logic includes:

- Missing value threshold checks  
- Statistical sensor drift detection  
- Logical bounds on critical sensors  

These checks are designed to execute prior to inference in production systems.

---

### 3.3 Data Versioning Strategy

Each dataset and model artifact is associated with:

- Generation timestamp  
- Feature schema  
- Model version  

This enables traceability, reproducibility, and safe retraining.

---

## 4. Problem Framing & Modeling Approach

### 4.1 Business Framing

The problem is framed as:

> *Estimate the current DPF soot load and recommend proactive regeneration actions to avoid operational failures.*

Rather than a binary classification, the system predicts **continuous soot load (%)**, allowing flexible thresholds based on vehicle type and fleet policy.

---

### 4.2 Target Definition

- **Primary target:** DPF soot load percentage  
- **Derived output:** Regeneration recommendation based on configurable thresholds  

---

### 4.3 Tradeoffs Considered

| Aspect | Decision |
|------|---------|
| False positives | Acceptable (minor fuel penalty) |
| False negatives | High cost (engine derate, downtime) |
| Early warning | Prioritized over perfect accuracy |
| Interpretability | Balanced with performance |

---

### 4.4 Modeling Choice

A **Random Forest Regressor** was selected because:

- Handles non-linear relationships between sensors  
- Robust to noisy data  
- Requires minimal preprocessing  
- Provides reasonable interpretability  

The primary evaluation metric is **Mean Absolute Error (MAE)**, which aligns better with operational cost than squared error metrics.

---

## 5. Evaluation Strategy

### Offline Evaluation

- MAE across historical data  
- Error inspection near critical soot thresholds (60–80%)  

### Production Evaluation

- Monitoring prediction distributions  
- Comparing predictions with post-regeneration outcomes  
- Tracking false alert rates  

Success is defined not just by model accuracy, but by **reduced unplanned maintenance events**.

---

## 6. Production & MLOps Considerations

### 6.1 Model Training & Artifact Management

- Reproducible training scripts  
- Hyperparameter logging  
- Versioned model artifacts  

---

### 6.2 Model Serving

A FastAPI-based service exposes:

- `/predict/soot-load`  
- `/predict/batch` (extendable)  
- `/model/info`  
- `/health`  

The API is stateless and suitable for scaling across fleets.

---

### 6.3 Containerization

The system is designed for Docker-based deployment with pinned dependencies, allowing cloud or edge deployment.

---

## 7. System Robustness & Edge Cases

Handled or considered scenarios include:

- Missing sensor readings  
- Out-of-range values  
- Cold-start vehicles with new DPFs  
- Immediately post-regeneration states  
- Delayed or stale data  

**Testing strategy includes:**
- Unit tests for feature logic  
- Integration tests for the full pipeline  
- Mock data simulations  

---

## 8. Business Impact of Prediction Errors

- **False positives:** Slight fuel penalty, acceptable  
- **False negatives:** Severe risk of downtime and component damage  

Therefore, the system intentionally biases toward **early warning**.

---

## 9. Conclusion & Recommendation

This project demonstrates a scalable and production-aware approach to DPF soot load prediction using vehicle telemetry data. With real-world data and feedback loops, the system can significantly reduce unplanned downtime and improve fleet efficiency.

# Deep Learning Optimization Techniques: A Comprehensive Guide

In deep learning, optimizing the training process is crucial for achieving accurate and efficient models. This guide provides an overview of essential techniques used to enhance the training process of neural networks, ranging from simple data preprocessing methods to advanced optimization algorithms.

These techniques are widely used in modern machine learning frameworks and can significantly affect the performance and convergence speed of your models.

---

## 📚 Topics Covered

### 1. **Feature Scaling**  
Feature scaling is a preprocessing technique that resizes input data to a standard range, ensuring that no feature dominates others and improving the performance of gradient-based optimization methods.

### 2. **Batch Normalization**  
Batch Normalization is a technique to stabilize and accelerate training by normalizing the activations of each layer, helping to reduce internal covariate shift.

### 3. **Mini-Batch Gradient Descent**  
Mini-Batch Gradient Descent strikes a balance between the computational efficiency of batch gradient descent and the stability of stochastic gradient descent, improving convergence speed.

### 4. **Gradient Descent with Momentum**  
Momentum accelerates convergence by adding a fraction of the previous update to the current one, helping the optimization process escape local minima and speed up the learning.

### 5. **RMSProp Optimization**  
RMSProp dynamically adjusts the learning rate for each parameter, normalizing gradients by their recent magnitudes to improve training stability and efficiency.

### 6. **Adam Optimization**  
Adam (Adaptive Moment Estimation) combines the benefits of momentum and RMSProp, adjusting the learning rate based on both first and second moment estimates, offering a powerful method for training complex models.

### 7. **Learning Rate Decay**  
Learning rate decay gradually reduces the learning rate over time to allow the model to settle into a minimum more smoothly, preventing overshooting and improving convergence.

---

## 📈 Why Optimization Matters

Training deep neural networks requires careful tuning of several hyperparameters. Poor optimization can lead to slow convergence, overfitting, or even failure to train. By applying these techniques, you can significantly improve the performance of your models, reduce training time, and achieve better results.

In the following sections, we'll dive deep into each of these topics, explaining the theory, formulas, and practical applications. Whether you're a beginner looking to understand the basics or an advanced practitioner refining your models, this guide will provide valuable insights for optimizing your deep learning workflows.



# Feature Scaling in Machine Learning: A Practical Guide

![Feature Scaling Visualization](https://datasciencedojo.com/wp-content/uploads/feature-scaling-techniques-1.webp)  
*Illustration of data distribution before (left) and after (right) feature scaling*

Feature scaling is a fundamental preprocessing step that significantly impacts the performance of many machine learning algorithms. Let's break down its mechanics, benefits, and limitations.

---

## What is Feature Scaling?

### Mechanics
Feature scaling standardizes/normalizes numerical features to a consistent scale. Two primary methods:



### 1. Min-Max Normalization
Rescales features to the [0, 1] range:

![Min-Max Formula](https://latex.codecogs.com/png.image?\dpi{120}X_{\text{norm}}=\frac{X-X_{\min}}{X_{\max}-X_{\min}})




### 2. Z-Score Standardization
Centers data around μ = 0 with σ = 1:

![Z-Score Formula](https://latex.codecogs.com/png.image?\dpi{120}X_{\text{std}}=\frac{X-\mu}{\sigma})





![Scaling Methods Comparison](https://miro.medium.com/v2/resize:fit:1400/1*y0esOCH8O2NV1c_8iY3ouA.png)  
*Comparison of normalization vs standardization effects*

---

## Why Use Feature Scaling?

### Pros ✅
1. **Faster Convergence**  
   Gradient-based algorithms (linear regression, neural networks) converge faster when features are scaled.

2. **Prevents Feature Dominance**  
   Ensures features with large ranges (e.g., 0-100000) don't overpower others (e.g., 0-1).

3. **Improved Distance Calculations**  
   Essential for distance-based algorithms:  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machines (SVM)  
   - K-Means Clustering

4. **Better Regularization**  
   Helps L1/L2 regularization treat all features equally.

---

## When to Avoid Feature Scaling

### Cons ❌
1. **Tree-Based Algorithms**  
   Decision Trees, Random Forests, and XGBoost don't require scaling as they split data regardless of scale.

2. **Sparse Data Challenges**  
   Can destroy sparsity in data (e.g., one-hot encoded features).

3. **Information Loss Risk**  
   Improper scaling might distort meaningful relative differences.

---

## Practical Implementation

### Python Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data: [price (USD), room_count]
X = np.array([[250000, 3], 
              [875000, 4], 
              [150000, 2]])

# Z-Score Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Min-Max Normalization
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)

print("Standardized:\n", X_standardized)
print("\nNormalized:\n", X_normalized)


```




# Batch Normalization in Deep Learning: The Stabilization Secret

![Batch Normalization Effect](https://gradientscience.org/images/batchnorm/vgg_bn_good_train.jpg)  
*Visualization of layer activations without (left) and with (right) batch normalization*

Batch normalization (BN) is a revolutionary technique that dramatically improves the training of deep neural networks. Let's explore its inner workings and practical implications.

---

## How Batch Normalization Works

### Mechanics 📐
BN normalizes layer outputs during training through three steps:

### 1. **Mini-Batch Statistics Calculation**  
For each mini-batch:

![Batch Mean](https://latex.codecogs.com/png.image?\dpi{120}\mu_B%20=%20\frac{1}{m}%20\sum_{i=1}^m%20x_i%20\quad%20\text{(batch%20mean)})

![Batch Variance](https://latex.codecogs.com/png.image?\dpi{120}\sigma_B^2%20=%20\frac{1}{m}%20\sum_{i=1}^m%20(x_i%20-%20\mu_B)^2%20\quad%20\text{(batch%20variance)})

---

### 2. **Normalization**  

![Normalization](https://latex.codecogs.com/png.image?\dpi{120}\hat{x}_i%20=%20\frac{x_i%20-%20\mu_B}{\sqrt{\sigma_B^2%20+%20\epsilon}}%20\quad%20\text{(ϵ%20=%20small%20safety%20constant)})

---

### 3. **Scale and Shift**  

![Scale and Shift](https://latex.codecogs.com/png.image?\dpi{120}y_i%20=%20\gamma%20\hat{x}_i%20+%20\beta%20\quad%20\text{(γ%20and%20β%20are%20learnable%20parameters)})


![BN Process](https://miro.medium.com/v2/resize:fit:898/0*pSSzicm1IH4hXOHc.png)  
*Batch normalization pipeline: Calculate statistics → Normalize → Scale/Shift*

---

## Why Use Batch Normalization?

### Pros ✅
1. **Faster Training**  
   Reduces internal covariate shift, allowing 2-5x higher learning rates.

2. **Regularization Effect**  
   Adds noise through mini-batch statistics, reducing overfitting.

3. **Reduces Sensitivity**  
   Makes networks less sensitive to weight initialization.

4. **Smother Gradients**  
   Prevents activation values from saturating in non-linear regions.

5. **Enables Deeper Networks**  
   Makes training of 100+ layer networks feasible.

---

## Challenges and Limitations

### Cons ❌
1. **Batch Size Dependency**  
   Performs poorly with very small batches (<8 samples)

2. **Increased Computation**  
   Adds ~10-20% training time overhead

3. **Inference Complexity**  
   Uses population statistics (EMA of batch stats) during prediction

4. **RNN Challenges**  
   Hard to apply to recurrent networks due to variable sequence lengths

---

## Practical Implementation

### Keras Code Example
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),  # ← BN layer after activation
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')


```




# Mini-Batch Gradient Descent: The Goldilocks of Optimization

![Mini-Batch Gradient Descent Diagram](https://statusneo.com/wp-content/uploads/2023/09/Credit-Analytics-Vidya.jpg)  
*Comparison of gradient descent variants: Batch (left), Mini-Batch (center), Stochastic (right)*

Mini-batch gradient descent strikes a perfect balance between computational efficiency and optimization stability, making it the most widely used optimization approach in modern machine learning. Let's explore its inner workings and practical considerations.

---

## How Mini-Batch GD Works

### Mechanics 🛠️
1. **Data Partitioning**  
   Split dataset into small batches (typically 32-512 samples)

2. **Iterative Updates**  
For each batch:

  ![Iterative Update](https://latex.codecogs.com/png.image?\dpi{120}\theta_{t+1}%20=%20\theta_t%20-%20\eta%20\cdot%20\frac{1}{m}%20\sum_{i=1}^m%20\nabla_\theta%20J(\theta;%20x^{(i)},%20y^{(i)}))

Where:
- ![eta](https://latex.codecogs.com/png.image?\dpi{120}\eta) = learning rate  
- ![m](https://latex.codecogs.com/png.image?\dpi{120}m) = batch size  
- ![theta](https://latex.codecogs.com/png.image?\dpi{120}\theta) = model parameters

3. **Epoch Completion**  
   Process all batches → 1 epoch

![Update Process](https://miro.medium.com/v2/resize:fit:754/1*mzPwfNdy6Yo0VTyDCh8S9A.png)  
*Parameter update workflow: Batch selection → Gradient calculation → Weight update*

---

## Why Mini-Batch GD Dominates Practice

### Pros ✅
| Advantage | Explanation |
|-----------|-------------|
| 🚀 **Computational Efficiency** | Leverages GPU parallelism better than pure SGD |
| 📉 **Stable Convergence** | Smoother updates than SGD (lower variance) |
| 💾 **Memory Friendly** | Processes large datasets without full RAM load |
| ⚖️ **Tradeoff Balance** | Combines Batch GD's stability with SGD's speed |
| 🔧 **Hardware Optimization** | Batch sizes often align with GPU memory boundaries |

---

## Challenges and Considerations

### Cons ❌
| Limitation | Mitigation Strategies |
|------------|-----------------------|
| **Batch Size Tuning** | Start with 32/64, then experiment |
| **Local Minima Risk** | Use momentum or Adam optimizer |
| **No True Minimum** | Implement learning rate decay |
| **Epoch Comparison** | Metrics become batch-dependent |

---

## Practical Implementation

### TensorFlow Code Example
```python
import tensorflow as tf

# Load dataset
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Configure mini-batches
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)

# Model and training
model = tf.keras.Sequential([...])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

# Mini-batch GD in action
model.fit(train_dataset, epochs=10)


```




# Gradient Descent with Momentum: The Physics-Inspired Optimizer

![Momentum Optimization Path](https://miro.medium.com/v2/resize:fit:1000/1*X9SaxFM6_sBOAMY9TaGsKw.png)  
*Comparison of standard GD (left) and momentum-enhanced GD (right) in a ravine-like loss landscape*

Momentum supercharges gradient descent by incorporating velocity into parameter updates, creating one of the most impactful optimization enhancements in deep learning. Let's explore this physics-inspired technique.

---

## The Mechanics of Momentum

### Core Concept 🌪️  
Momentum accumulates past gradients to determine update direction:

![Momentum Velocity](https://latex.codecogs.com/png.image?\dpi{120}v_t%20=%20\beta%20v_{t-1}%20+%20(1%20-%20\beta)%20\nabla_\theta%20J(\theta_t))

![Momentum Update](https://latex.codecogs.com/png.image?\dpi{120}\theta_{t+1}%20=%20\theta_t%20-%20\eta%20v_t)

Where:
- ![beta](https://latex.codecogs.com/png.image?\dpi{120}\beta) = momentum coefficient (typically 0.9)  
- ![eta](https://latex.codecogs.com/png.image?\dpi{120}\eta) = learning rate  
- ![vt](https://latex.codecogs.com/png.image?\dpi{120}v_t) = velocity vector


### Physical Analogy
Imagine a ball rolling downhill:
- Gradient → Slope steepness
- Momentum → Ball's inertia
- β → Friction coefficient

![Momentum Dynamics](https://i.sstatic.net/gjDzm.gif)  
*Visualization of current gradient (red) and momentum-accumulated direction (blue)*

---

## Why Momentum Works

### Pros ✅
| Advantage | Impact |
|-----------|--------|
| 🚀 **Ravine Navigation** | Accelerates along shallow, stable directions |
| 📉 **Oscillation Damping** | Smoothens zig-zag paths in steep dimensions |
| 💨 **Escape Local Minima** | Momentum carries updates through flat regions |
| ⏱️ **Faster Convergence** | Often 2-10x speedup over vanilla GD |

### Performance Comparison
| Scenario | Vanilla GD Steps | Momentum GD Steps |
|----------|------------------|-------------------|
| Steep ravine | 1500 | 220 |
| Gentle slope | 450 | 300 |
| Noisy terrain | 1200 | 600 |

---

## Challenges and Limitations

### Cons ❌
| Limitation | Mitigation Strategy |
|------------|---------------------|
| **Hyperparameter Sensitivity** | Start with β=0.9, η=0.01 |
| **Overshooting Risk** | Combine with learning rate decay |
| **Velocity Initialization** | Zero-initialize velocity vectors |
| **Non-Convex Landscapes** | Use adaptive methods (Adam) instead |

---

## Practical Implementation

### TensorFlow Code Example
```python
from tensorflow.keras.optimizers import SGD

# Momentum optimizer configuration
optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,  # ← Momentum parameter
    nesterov=False  # Standard momentum
)

model.compile(optimizer=optimizer, loss='mse')


```




# RMSProp Optimization: Adaptive Learning Rate Mastery

![RMSProp Optimization Path](https://media.datacamp.com/cms/google/ad_4nxeo_f0dgwj0q84hnqy0la11q6kogdz8k2cc4vuldiaasbq-bvlo09zpleow-hv8arfjitglukefdiqhy7f4tghtpterr6maipfc8gw7dhswawm47veulhplnnafxzzfjnw50xtdfiuavub8axbnmopwxnt4.png)  
*Comparison of RMSProp (right) and SGD (left) navigating a non-convex loss landscape*

RMSProp (Root Mean Square Propagation) revolutionized adaptive learning rate optimization by introducing per-parameter scaling. Let's dissect this elegant algorithm that powers many modern deep learning systems.

---

## The RMSProp Algorithm Explained

### Core Mechanics 🔧

RMSProp dynamically adjusts learning rates using an exponential moving average of squared gradients:

### 1. **Accumulate Squared Gradients**  

![Accumulate Gradients](https://latex.codecogs.com/png.image?\dpi{120}E[g^2]_t%20=%20\gamma%20E[g^2]_{t-1}%20+%20(1%20-%20\gamma)%20g_t^2)

---

### 2. **Parameter Update**  

![Param Update](https://latex.codecogs.com/png.image?\dpi{120}\theta_{t+1}%20=%20\theta_t%20-%20\frac{\eta}{\sqrt{E[g^2]_t%20+%20\epsilon}}%20g_t)

---

Where:  
- ![gamma](https://latex.codecogs.com/png.image?\dpi{120}\gamma) = decay rate (typically 0.9)  
- ![eta](https://latex.codecogs.com/png.image?\dpi{120}\eta) = base learning rate  
- ![epsilon](https://latex.codecogs.com/png.image?\dpi{120}\epsilon) = smoothing constant (~1e-7)


![RMSProp Visualization](https://ml-explained.com/articles/rmsprop-explained/rmsprop_example.PNG)  
*Visualization of per-parameter learning rate scaling*

---

## Why RMSProp Works

### Key Advantages ✅

| Advantage | Impact | Use Case |
|-----------|--------|----------|
| **Non-Stationary Handling** | Automatically adapts to changing gradients | Time-series data |
| **Sparse Gradient Mastery** | Maintains appropriate step sizes | NLP tasks |
| **Ravine Navigation** | Prevents oscillations in steep dimensions | Deep CNNs |
| **Scale Invariance** | Equal treatment of all features | Mixed-scale data |

### Performance Benchmarks
| Metric | SGD | RMSProp |
|--------|-----|---------|
| Time to convergence | 4.2h | 1.8h |
| Final accuracy | 92.1% | 93.6% |
| Batch variance | High | Low |

---

## Limitations and Challenges

### Cons ❌

| Limitation | Mitigation Strategy |
|------------|---------------------|
| **No Momentum** | Combine with Nesterov momentum |
| **Hyperparameter Sensitivity** | Use default γ=0.9, ε=1e-7 |
| **Cold Start** | Warm-up phase helps initialization |
| **Stationary Gradient Issues** | Switch to Adam for stability |

---

## Practical Implementation

### TensorFlow Code Example
```python
from tensorflow.keras.optimizers import RMSprop

model = tf.keras.Sequential([...])

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9,          # γ (decay rate)
    epsilon=1e-07,    # Smoothing term
    centered=False     # Standard RMSProp
)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')


```




# Adam Optimization: The Adaptive Moment Estimation Powerhouse

![Adam Optimization Path](https://miro.medium.com/v2/resize:fit:1200/1*STiRp7PW5yIrvYZupZA6nw.gif)  
*Visualization of Adam's efficient pathfinding (gold) vs SGD (blue) and RMSProp (green)*

Adam (Adaptive Moment Estimation) has become the default optimizer for many deep learning tasks, combining the best features of momentum and adaptive learning rate methods. Let's explore why it's so widely adopted and how to use it effectively.

---

## Adam Algorithm Mechanics

### Core Equations 🧮

### 1. **Moment Estimates**  

**First moment (mean):**  
![m_t](https://latex.codecogs.com/png.image?\dpi{120}m_t%20=%20\beta_1%20m_{t-1}%20+%20(1%20-%20\beta_1)%20g_t)

**Second moment (variance):**  
![v_t](https://latex.codecogs.com/png.image?\dpi{120}v_t%20=%20\beta_2%20v_{t-1}%20+%20(1%20-%20\beta_2)%20g_t^2)

---

### 2. **Bias Correction**  

![m_hat](https://latex.codecogs.com/png.image?\dpi{120}\hat{m}_t%20=%20\frac{m_t}{1%20-%20\beta_1^t})

![v_hat](https://latex.codecogs.com/png.image?\dpi{120}\hat{v}_t%20=%20\frac{v_t}{1%20-%20\beta_2^t})

---

### 3. **Parameter Update**  

![theta_update](https://latex.codecogs.com/png.image?\dpi{120}\theta_{t+1}%20=%20\theta_t%20-%20\frac{\eta}{\sqrt{\hat{v}_t}%20+%20\epsilon}%20\hat{m}_t)

---

Where:  
- ![beta_1](https://latex.codecogs.com/png.image?\dpi{120}\beta_1) = 0.9 (1st moment decay)  
- ![beta_2](https://latex.codecogs.com/png.image?\dpi{120}\beta_2) = 0.999 (2nd moment decay)  
- ![epsilon](https://latex.codecogs.com/png.image?\dpi{120}\epsilon) = 1e-8 (numerical stability)


![Adam Dynamics](https://mdrk.io/content/images/2024/04/adam_optimization_trajectory.png)  
*Combination of momentum (blue) and adaptive learning rates (orange) in Adam*

---

## Why Adam Dominates Deep Learning

### Key Advantages ✅

| Advantage | Impact | Use Case |
|-----------|--------|----------|
| **Automatic Learning Rates** | Per-parameter adaptation | High-dimensional spaces |
| **Momentum Integration** | Smooths noisy gradients | Natural language processing |
| **Bias Correction** | Better early training | Few-shot learning |
| **Low Memory Footprint** | Only 2x parameters | Large models |
| **Default Performance** | Works well out-of-box | Rapid prototyping |

### Performance Benchmarks
| Dataset | SGD Time | Adam Time | Accuracy Boost |
|---------|----------|-----------|----------------|
| ImageNet | 72h | 68h | +1.2% |
| WikiText | 12h | 9h | +0.8 PPL |
| MNIST | 45m | 32m | +0.4% |

---

## Limitations and Considerations

### Cons ❌

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| **Generalization Gap** | Switch to SGD for final tuning |
| **Hyperparameter Sensitivity** | Keep default β values |
| **Memory Overhead** | Use gradient checkpointing |
| **Local Minima Escape** | Combine with SWA |

---

## Practical Implementation

### Keras Example
```python
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW

model = TransformerModel()
optimizer = AdamW(
    learning_rate=3e-4,     # Default learning rate
    beta_1=0.9,             # β1 and β2
    beta_2=0.999,
    epsilon=1e-8,
    weight_decay=0.01       # L2 regularization
)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = your_loss_function(y_true, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


```




# Learning Rate Decay: The Art of Fine-Tuning Optimization

![Learning Rate Decay Schedules](https://miro.medium.com/v2/resize:fit:1200/1*o6NiajP5MAcKxLas__NueQ.png)  
*Visualization of different decay strategies over training epochs*

Learning rate decay is a critical technique for balancing speed and precision in neural network training. This guide explores its mechanics, benefits, and practical implementation.

---

## How Learning Rate Decay Works

### Core Concept 📉  
Gradually reduce the learning rate (![eta](https://latex.codecogs.com/png.image?\dpi{120}\eta)) during training:

![LR Decay](https://latex.codecogs.com/png.image?\dpi{120}\eta_t%20=%20\text{Initial%20Rate}%20\times%20\text{Decay%20Function}(t))


### Common Decay Strategies

| Type             | Formula                                                                                                                                     | Behavior                          |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| **Step Decay**    | ![Step Decay](https://latex.codecogs.com/png.image?\dpi{120}\eta_t%20=%20\eta_0%20\times%20\gamma^{\lfloor%20t/s%20\rfloor})                | Drops η by factor γ every s steps |
| **Exponential**   | ![Exponential Decay](https://latex.codecogs.com/png.image?\dpi{120}\eta_t%20=%20\eta_0%20\times%20e^{-kt})                                  | Smooth exponential decline        |
| **Time-Based**    | ![Time-Based Decay](https://latex.codecogs.com/png.image?\dpi{120}\eta_t%20=%20\frac{\eta_0}{1%20+%20kt})                                   | Gentle reciprocal decay           |
| **Cosine**        | ![Cosine Decay](https://latex.codecogs.com/png.image?\dpi{120}\eta_t%20=%20\frac{\eta_0}{2}(1%20+%20\cos(\pi%20t/T)))                        | Wave-like reduction               |



---

## Why Use Learning Rate Decay?

### Pros ✅
1. **Precision Refinement**  
   Enables sub-pixel accuracy near minima

2. **Training Stability**  
   Reduces loss oscillations in later stages

3. **Escape Saddle Points**  
   Temporary rate spikes can kick models out of plateaus

4. **Hyperparameter Flexibility**  
   Works with any optimizer (SGD, Adam, etc.)

### Performance Impact
| Metric | Fixed η | Decaying η |
|--------|---------|------------|
| Final Accuracy | 91.3% | 93.1% |
| Training Time | 2.1h | 1.8h |
| Loss Variance | 0.32 | 0.11 |

---

## Challenges and Limitations

### Cons ❌
| Challenge | Solution |
|-----------|----------|
| **Premature Decay** | Add warmup phase |
| **Schedule Tuning** | Use validation-based auto-decay |
| **Adaptive Conflict** | Reduce decay strength with Adam |
| **Global vs Local** | Layer-specific decay (advanced) |

---

## Practical Implementation

### TensorFlow Example
```python
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy')




# Обучение с подкреплением (Ответы на вопросы)

## Содержание
- [Неделя №1](#неделя-1)
- [Неделя №2](#неделя-2)
- [Неделя №3](#неделя-3)
- [Неделя №4](#неделя-4)
- [Неделя №6](#неделя-6)
- [Неделя №7](#неделя-7)

---

## Неделя №1

### 1.1. Базовые понятия обучения с подкреплением

**Обучение с подкреплением (RL)** — метод машинного обучения, при котором агент учится принимать решения, взаимодействуя со средой методом проб и ошибок, получая награды или штрафы.

**Области применения**: интернет-маркетинг, робототехника, игры, медицина, финансы, диалоговые системы и др.

**Основные сущности**:
- **Агент** — принимает решения.
- **Среда** — внешний мир, выдаёт состояния и награды.

**Цикл взаимодействия**:
- **Действие** (Action, \(a\))
- **Состояние** (State, \(s\)) — обладает марковским свойством.
- **Награда** (Reward, \(r\))
- **Стратегия** (Policy, \(\pi(a|s)\)) — правило выбора действия.

Стратегия задаётся как условная вероятность:

```math
\pi(a|s) = P(\text{совершить действие } a \mid \text{находимся в состоянии } s)
```

**Цели агента**:
- Максимизировать суммарное ожидаемое вознаграждение за эпизод.
- **Эпизод** — последовательность:

```math
(s_0, a_0, r_0, s_1, a_1, r_1, \dots)
```

- Суммарная награда:

```math
R = \sum_{i} r_i
```

- Цель обучения:

```math
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[R]
```

**Отличие от других парадигм**:
- Обучение с учителем — есть готовые ответы.
- Обучение без учителя — поиск структур без обратной связи.
- RL — нет эталонов, действия влияют на будущие данные, награда — ключевой сигнал.

---

### 1.2. Метод кросс-энтропии (CEM)

Итеративный алгоритм, основанный на отборе лучших стратегий.

Цикл:
1. Проигрывание \(N\) сессий.
2. Выбор \(M\) лучших («элитных»).
3. Обновление стратегии, чтобы успешные действия стали более вероятными.

#### 1.2.1. Табличный метод

Применяется при небольшом числе состояний и действий.

- Стратегия — матрица вероятностей:

```math
\pi(a|s) = A_{a,s}
```

(сумма по действиям для каждого состояния равна 1).

- Сбор данных: \(N\) сессий.
- Отбор элитных: сохраняем все пары \((s,a)\) из \(M\) лучших сессий.

```math
\text{Elite} = [(s_0, a_0), (s_1, a_1), \dots]
```

- Обновление стратегии:

```math
\pi(a|s) = \frac{\text{число выборов } a \text{ в состоянии } s \text{ в элитных сессиях}}{\text{число посещений состояния } s \text{ в элитных сессиях}}
```

- Критерий остановки — по числу итераций или стабильности политики.

#### 1.2.2. Аппроксимированный метод

При большом пространстве состояний используется нейросеть \(f_W(s)\), предсказывающая вероятности действий.

- Инициализация весов \(W\) случайно.
- Сбор \(N\) сессий.
- Отбор элитных (\(M\)).
- Оптимизация — максимизация логарифма правдоподобия действий из элитных сессий:

```math
W_{i+1} = W_i + \alpha \nabla \sum_{(s_i, a_i) \in \text{Elite}} \log \pi_{W_i}(a_i|s_i)
```

- Для непрерывных действий стратегию моделируют как гауссово распределение:

```math
\pi_W(a|s) = \mathcal{N}(\mu(s), \sigma^2)
```

---

## Неделя №2

### 2.1. V(s) и Q(s,a)

**Функция ценности состояния** \(V_\pi(s)\) — ожидаемая дисконтированная сумма наград, начиная с состояния \(s\) и далее следуя политике \(\pi\):

```math
V_\pi(s) \triangleq \mathbb{E}_\pi[G_t \mid S_t = s]
```

где дисконтированная награда:

```math
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad 0 \le \gamma < 1
```

\(\gamma\) — коэффициент дисконтирования.

Рекуррентная форма:

```math
G_t = R_t + \gamma G_{t+1}
```

Уравнение Беллмана для \(V_\pi\):

```math
V_\pi(s) = \sum_a \pi(a|s) \sum_{r,s'} p(r,s'|s,a) \big[ r + \gamma V_\pi(s') \big]
```

**Функция ценности действия** \(Q_\pi(s,a)\):

```math
Q_\pi(s,a) = \sum_{r,s'} p(r,s'|s,a) \big[ r + \gamma V_\pi(s') \big]
```

Оптимальная политика \(\pi^*\) даёт максимальные \(V\) и \(Q\) во всех состояниях.

---

### 2.2. Метод Policy Iteration

Алгоритм поиска оптимальной стратегии в MDP.

1. **Инициализация**: произвольная \(\pi\), \(V(s)=0\).
2. **Оценка политики** — итеративно решаем уравнение Беллмана для фиксированной \(\pi\):

```math
V(s) = \sum_{s',r} p(s',r|s,\pi(s)) \big[ r + \gamma V(s') \big]
```

Повторяем до сходимости.

3. **Улучшение политики** — делаем стратегию жадной относительно \(V\):

```math
\pi(s) = \arg\max_a \sum_{s',r} p(s',r|s,a) \big[ r + \gamma V(s') \big]
```

Если \(\pi\) не изменилась — останов, иначе повторяем с шага 2.

**Сравнение с Value Iteration**:
- Policy Iteration — полная оценка на каждом шаге, но меньше внешних итераций.
- Value Iteration — одна итерация оценки, но больше внешних шагов.

---

## Неделя №3

### 3.1. Q-learning и SARSA

Оба — model-free методы семейства TD, обучают \(Q(s,a)\).

#### Q-learning (off-policy)

Обновление использует максимальное Q-значение следующего состояния:

```math
Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha \big( r + \gamma \max_{a'} Q(s',a') \big)
```

#### SARSA (on-policy)

Учитывает реальное следующее действие \(a'\):

```math
Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha \big( r + \gamma Q(s',a') \big)
```

---

### 3.2. Exploration vs Exploitation

**Дилемма**:
- **Exploitation** — выбор наилучшего по текущим оценкам действия.
- **Exploration** — исследование новых действий.

Решения:
- \(\varepsilon\)-greedy: с вероятностью \(\varepsilon\) случайное действие, иначе жадное.
- **Softmax** — выбор пропорционален экспоненте:

```math
\pi(a|s) = \frac{\exp(Q(s,a)/T)}{\sum_{a'} \exp(Q(s,a')/T)}
```

где \(T\) — температура.

---

## Неделя №4

### 4.1. DQN (Deep Q-Network)

Заменяет таблицу \(Q(s,a)\) нейросетью, которая по состоянию \(s\) выдаёт вектор Q-значений для всех действий.

**Проблемы обучения** и их решения:
- **Experience Replay** — буфер переходов \((s,a,r,s')\).
- **Target Network** — отдельная сеть \(\Theta^-\) для стабильных целей.

Функция потерь:

```math
L(\Theta) = \mathbb{E}_{(s,a,r,s') \sim \text{Buffer}} \left[ \big( Q(s,a,\Theta) - (r + \gamma \max_{a'} Q(s',a',\Theta^-)) \big)^2 \right]
```

---

### 4.2. Experience Replay

- Сохраняем до \(10^6\) последних переходов.
- Сэмплируем батчи случайно.
- Решает проблемы корреляции, забывания, неэффективности использования опыта.

---

### 4.3. Target Network

Целевая сеть обновляется реже.

Способы обновления:
- **Hard update**: \(\Theta^- \leftarrow \Theta\) каждые \(n\) шагов.
- **Soft update**:

```math
\Theta^- \leftarrow (1-\tau)\Theta^- + \tau\Theta, \quad \tau \ll 1
```

---

### 4.4. Double DQN (DDQN)

Исправляет завышение оценок:

- Выбор действия по основной сети:

```math
a^* = \arg\max_{a'} Q(s', a', \Theta)
```

- Оценка по целевой сети:

```math
\text{Target} = r + \gamma \, Q(s', a^*, \Theta^-)
```

Итоговая формула:

```math
r + \gamma \, Q(s',\, \arg\max_{a'} Q(s',a',\Theta),\, \Theta^-)
```

---

## Неделя №6

### 5.1. Policy Gradient

Методы, оптимизирующие стратегию \(\pi_\theta(a|s)\) напрямую.

**Целевая функция**:

```math
J(\theta) = \mathbb{E}_{s \sim p, a \sim \pi_\theta} [ G(s,a) ]
```

**Градиент** (с использованием логарифмической производной):

```math
\nabla J(\theta) = \mathbb{E}\big[ \nabla_\theta \log \pi_\theta(a|s) \, Q(s,a) \big]
```

---

### 5.2. REINFORCE

Базовый on-policy алгоритм.

1. Инициализация \(\theta\).
2. Генерация \(N\) сессий.
3. Оценка градиента:

```math
\nabla J \approx \frac{1}{N} \sum_{i} \sum_{s,a} \nabla \log \pi_\theta(a|s) \, Q(s,a)
```

4. Обновление:

```math
\theta \leftarrow \theta + \alpha \nabla J
```

**Снижение дисперсии** с помощью базовой линии \(b(s)\) (часто \(V(s)\)):

```math
\nabla J \approx \frac{1}{N} \sum_{i} \sum_{s,a} \nabla \log \pi_\theta(a|s) \, (Q(s,a) - b(s))
```

---

### 5.3. Actor-Critic

Гибрид: **Actor** — стратегия, **Critic** — оценка ценности.

В **Advantage Actor-Critic (A2C)**:

```math
A(s,a) = r + \gamma V(s') - V(s)
```

Обучение:
- Actor выбирает действие.
- Critic вычисляет TD-ошибку \(\delta = r + \gamma V(s') - V(s)\).
- Critic минимизирует MSE.
- Actor обновляется, увеличивая вероятность действий с положительным \(\delta\).

---

## Неделя №7

### 6.1. Применение RNN для генерации последовательностей

Архитектура **Encoder-Decoder**:
- **Encoder** — сжимает входную последовательность в вектор.
- **Decoder** — генерирует выходную последовательность пошагово.

**Области применения**: машинный перевод, генерация описаний изображений, диалоговые системы и др.

Обучение — с учителем, цель — максимизация логарифма вероятности эталонного ответа.

**Проблема**: сдвиг распределения (distribution shift) — модель использует свои собственные предсказания на этапе генерации.

---

### 6.2. Применение RL к дообучению моделей генерации

RL используется для устранения сдвига распределения и оптимизации недифференцируемых метрик.

**Основные подходы**:

- **Self-Critical Sequence Training (SCST)** — базовая линия — результат жадной генерации.
- **RLHF (Reinforcement Learning from Human Feedback)** — три этапа: SFT, обучение модели вознаграждения, оптимизация с PPO.
- **DPO (Direct Preference Optimization)** — оптимизация напрямую на предпочтениях без отдельной модели вознаграждения.
















# Practical_RL

An open course on reinforcement learning in the wild.
Taught on-campus at [HSE](https://cs.hse.ru) and [YSDA](https://yandexdataschool.com/)  and maintained to be friendly to online students (both english and russian).


#### Manifesto:
* __Optimize for the curious.__ For all the materials that aren’t covered in detail there are links to more information and related materials (D.Silver/Sutton/blogs/whatever). Assignments will have bonus sections if you want to dig deeper.
* __Practicality first.__ Everything essential to solving reinforcement learning problems is worth mentioning. We won't shun away from covering tricks and heuristics. For every major idea there should be a lab that makes you to “feel” it on a practical problem.
* __Git-course.__ Know a way to make the course better? Noticed a typo in a formula? Found a useful link? Made the code more readable? Made a version for alternative framework? You're awesome! [Pull-request](https://help.github.com/articles/about-pull-requests/) it!

[![Github contributors](https://img.shields.io/github/contributors/yandexdataschool/Practical_RL.svg?logo=github&logoColor=white)](https://github.com/yandexdataschool/Practical_RL/graphs/contributors)

# Course info

* __FAQ:__ [About the course](https://github.com/yandexdataschool/Practical_RL/wiki/Practical-RL), [Technical issues thread](https://github.com/yandexdataschool/Practical_RL/issues/1), [Lecture Slides](https://yadi.sk/d/loPpY45J3EAYfU), [Online Student Survival Guide](https://github.com/yandexdataschool/Practical_RL/wiki/Online-student's-survival-guide)

* Anonymous [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSdurWw97Sm9xCyYwC8g3iB5EibITnoPJW2IkOVQYE_kcXPh6Q/viewform).

* Virtual course environment: 
    * [__Google Colab__](https://colab.research.google.com/) - set open -> github -> yandexdataschool/pracical_rl -> {branch name} and select any notebook you want.
    * [Installing dependencies](https://github.com/yandexdataschool/Practical_RL/issues/1) on your local machine (recommended).
    * Alternative: [Azure Notebooks](https://notebooks.azure.com/).


# Additional materials
* [RL reading group](https://github.com/yandexdataschool/Practical_RL/wiki/RL-reading-group)


# Syllabus

The syllabus is approximate: the lectures may occur in a slightly different order and some topics may end up taking two weeks.

* [__week01_intro__](./week01_intro) Introduction
  * Lecture: RL problems around us. Decision processes. Stochastic optimization, Crossentropy method. Parameter space search vs action space search.
  * Seminar: Welcome into openai gym. Tabular CEM for Taxi-v0, deep CEM for box2d environments.
  * Homework description - see week1/README.md. 

* [__week02_value_based__](./week02_value_based) Value-based methods
  * Lecture: Discounted reward MDP. Value-based approach. Value iteration. Policy iteration. Discounted reward fails.
  * Seminar: Value iteration.  
  * Homework description - see week2/README.md. 
  
* [__week03_model_free__](./week03_model_free) Model-free reinforcement learning
  * Lecture: Q-learning. SARSA. Off-policy Vs on-policy algorithms. N-step algorithms. TD(Lambda).
  * Seminar: Qlearning Vs SARSA Vs Expected Value SARSA
  * Homework description - see week3/README.md. 

* [__recap_deep_learning__](./week04_\[recap\]_deep_learning) - deep learning recap 
  * Lecture: Deep learning 101
  * Seminar: Intro to pytorch/tensorflow, simple image classification with convnets

* [__week04_approx_rl__](./week04_approx_rl) Approximate (deep) RL
  * Lecture: Infinite/continuous state space. Value function approximation. Convergence conditions. Multiple agents trick; experience replay, target networks, double/dueling/bootstrap DQN, etc.
  * Seminar:  Approximate Q-learning with experience replay. (CartPole, Atari)
  
* [__week05_explore__](./week05_explore) Exploration
  * Lecture: Contextual bandits. Thompson Sampling, UCB, bayesian UCB. Exploration in model-based RL, MCTS. "Deep" heuristics for exploration.
  * Seminar: bayesian exploration for contextual bandits. UCB for MCTS.

* [__week06_policy_based__](./week06_policy_based) Policy Gradient methods
  * Lecture: Motivation for policy-based, policy gradient, logderivative trick, REINFORCE/crossentropy method, variance reduction(baseline), advantage actor-critic (incl. GAE)
  * Seminar: REINFORCE, advantage actor-critic

* [__week07_seq2seq__](./week07_seq2seq) Reinforcement Learning for Sequence Models
  * Lecture: Problems with sequential data. Recurrent neural networks. Backprop through time. Vanishing & exploding gradients. LSTM, GRU. Gradient clipping
  * Seminar: character-level RNN language model

* [__week08_pomdp__](./week08_pomdp) Partially Observed MDP
  * Lecture: POMDP intro. POMDP learning (agents with memory). POMDP planning (POMCP, etc)
  * Seminar: Deep kung-fu & doom with recurrent A3C and DRQN
  
* [__week09_policy_II__](./week09_policy_II) Advanced policy-based methods
  * Lecture: Trust region policy optimization. NPO/PPO. Deterministic policy gradient. DDPG
  * Seminar: Approximate TRPO for simple robot control.

* [__week10_planning__](./week10_planning) Model-based RL & Co
  * Lecture: Model-Based RL, Planning in General, Imitation Learning and Inverse Reinforcement Learning
  * Seminar: MCTS for toy tasks

* [__yet_another_week__](./yet_another_week) Inverse RL and Imitation Learning
  * All that cool RL stuff that you won't learn from this course :)


# Course staff
Course materials and teaching by: _[unordered]_
- [Pavel Shvechikov](https://github.com/pshvechikov) - lectures, seminars, hw checkups, reading group
- [Nikita Putintsev](https://github.com/qwasser) - seminars, hw checkups, organizing our hot mess
- [Alexander Fritsler](https://github.com/Fritz449) - lectures, seminars, hw checkups
- [Oleg Vasilev](https://github.com/Omrigan) - seminars, hw checkups, technical support
- [Dmitry Nikulin](https://github.com/pastafarianist) - tons of fixes, far and wide
- [Mikhail Konobeev](https://github.com/MichaelKonobeev) - seminars, hw checkups
- [Ivan Kharitonov](https://github.com/neer201) - seminars, hw checkups
- [Ravil Khisamov](https://github.com/zshrav) - seminars, hw checkups
- [Anna Klepova](https://github.com/q0o0p) - hw checkups
- [Fedor Ratnikov](https://github.com/justheuristic) - admin stuff

# Contributions
* Using pictures from [Berkeley AI course](http://ai.berkeley.edu/home.html)
* Massively refering to [CS294](http://rll.berkeley.edu/deeprlcourse/)
* Several tensorflow assignments by [Scitator](https://github.com/Scitator)
* A lot of fixes from [arogozhnikov](https://github.com/arogozhnikov)
* Other awesome people: see github [contributors](https://github.com/yandexdataschool/Practical_RL/graphs/contributors)
* [Alexey Umnov](https://github.com/alexeyum) helped us a lot during spring2018


# Обучение с подкреплением (Ответы на вопросы)

## Содержание

- [Неделя №1](#неделя-1)
  - [1.1. Базовые понятия обучения с подкреплением](#11-базовые-понятия-обучения-с-подкреплением)
  - [1.2. Метод кросс-энтропии](#12-метод-кросс-энтропии)
    - [1.2.1. Табличный метод кросс-энтропии](#121-табличный-метод-кросс-энтропии)
    - [1.2.2. Аппроксимированный метод кросс-энтропии](#122-аппроксимированный-метод-кросс-энтропии)
- [Неделя №2](#неделя-2)
  - [2.1. V(s), Q(s, a) – что это такое?](#21-vs-qsa--что-это-такое)
  - [2.2. Метод Policy Iteration](#22-метод-policy-iteration)
- [Неделя №3](#неделя-3)
  - [3.1. Q-learning, SARSA](#31-q-learning-sarsa)
  - [3.2. Exploration vs Exploitation](#32-exploration-vs-exploitation)
- [Неделя №4](#неделя-4)
  - [4.1. DQN](#41-dqn)
  - [4.2. Experience Replay](#42-experience-replay)
  - [4.3. Target Network](#43-target-network)
  - [4.4. Double DQN](#44-double-dqn)
- [Неделя №6](#неделя-6)
  - [5.1. Policy Gradient](#51-policy-gradient)
  - [5.2. REINFORCE](#52-reinforce)
  - [5.3. Actor-critic](#53-actor-critic)
- [Неделя №7](#неделя-7) *(добавлено)*
  - [6.1. Применение RNN для генерации последовательностей](#61-применение-rnn-для-генерации-последовательностей)
  - [6.2. Применение RL к дообучению моделей генерации последовательности](#62-применение-rl-к-дообучению-моделей-генерации-последовательности)

---

## Неделя №1

### 1.1. Базовые понятия обучения с подкреплением

**Обучение с подкреплением** (Reinforcement Learning, RL) — это метод машинного обучения, при котором алгоритм (агент) учится принимать оптимальные решения, взаимодействуя с окружающей средой методом проб и ошибок. Агент получает награды или штрафы за свои действия, стремясь максимизировать суммарное вознаграждение.

#### Области применения RL

- Интернет-маркетинг (выбор релевантных объявлений, рекомендательные системы, оптимизация дизайна целевых страниц).
- Робототехника и автономные системы (обучение ходьбе, беспилотные автомобили, помощь пилотам).
- Игровая индустрия.
- Персонализированная медицина (подбор стратегий лечения).
- Финансы (управление портфелями).
- Разговорные системы (обучение чат-ботов).
- Глубокое обучение (поиск архитектур нейросетей, оптимизация функций потерь).

#### Процесс взаимодействия

- **Агент (agent)** – алгоритм, принимающий решения и совершающий действия.
- **Среда (environment)** – внешний мир, в котором действует агент. Агент не знает её модели, а получает только состояния и награды в ответ на свои действия.

Циклическое взаимодействие:

- **Действие (Action, a)** – агент совершает шаг (например, показывает баннер пользователю).
- **Состояние (State, s)** – среда сообщает информацию о текущей ситуации (характеристики пользователя). Важно: вероятность перехода в следующее состояние зависит только от текущего состояния и действия (Марковское свойство).
- **Награда (Reward, r)** – числовая обратная связь от среды (например, кликнул ли пользователь на баннер).
- **Стратегия (Policy, π(a|s))** – правило выбора действия в состоянии `s`. Обычно задаётся как условная вероятность:

\[
\pi(a|s) = P(\text{совершить действие } a \mid \text{в состоянии } s).
\]

#### Цели и дополнительные понятия

- **Цель агента** – максимизировать не текущую награду, а **суммарное ожидаемое вознаграждение** за весь эпизод.
- **Эпизод (сессия)** – последовательность состояний, действий и наград:  
  \((s_0, a_0, r_0, s_1, a_1, r_1, \dots)\).
- **Суммарная награда** – \(R = \sum_i r_i\).
- **Цель обучения** – найти политику \(\pi(a|s)\), максимизирующую ожидаемую суммарную награду:  
  \(\mathbb{E}_\pi[R] \to \max\).

#### Отличие RL от других парадигм

- **Обучение с учителем** – алгоритм учится на готовых примерах с правильными ответами; модель не влияет на входные данные.
- **Обучение без учителя** – поиск скрытых структур в данных без обратной связи.
- **RL** – нет эталонных ответов; действия агента напрямую влияют на будущие данные; критически важна награда от среды.

---

### 1.2. Метод кросс-энтропии

**Метод кросс-энтропии** (Cross-Entropy Method, CEM) – итеративный алгоритм RL, основанный на отборе лучших стратегий. Цикл повторяет три этапа:

1. Проигрывание нескольких сессий.
2. Выбор лучших («элитных») сессий.
3. Обновление стратегии так, чтобы успешные действия стали приоритетными.

#### 1.2.1. Табличный метод кросс-энтропии

Применяется в задачах с небольшим количеством состояний и действий.

- **Представление стратегии**: матрица вероятностей \(\pi(a|s) = A_{a,s}\), где сумма по действиям для каждого состояния равна 1. Инициализируется случайно.
- **Сбор данных**: агент проигрывает \(N\) сессий.
- **Отбор элитных сессий**: выбираются \(M\) сессий с наибольшей суммарной наградой. Сохраняются все пары «состояние-действие» из этих сессий:  
  \(\text{Elite} = [(s_0, a_0), (s_1, a_1), \dots]\).
- **Обновление стратегии**: новая вероятность действия \(a\) в состоянии \(s\) вычисляется как отношение числа выборов \(a\) в состоянии \(s\) в элитных сессиях к общему числу посещений состояния \(s\) в этих сессиях:

\[
\pi(a|s) = \frac{\text{сколько раз выбрали } a \text{ в состоянии } s \text{ в элитных сессиях}}{\text{сколько раз были в состоянии } s \text{ в элитных сессиях}}
\]

- **Критерий остановки**: ограничение на количество итераций или изменение политики.

#### 1.2.2. Аппроксимированный метод кросс-энтропии

Когда пространство состояний слишком велико для таблицы, стратегию аппроксимируют с помощью нейронной сети (или линейной модели), которая принимает на вход состояние \(s\) и выдаёт вероятности действий.

- **Инициализация**: веса нейронной сети \(W\) задаются случайно.
- **Сбор данных**: агент проигрывает \(N\) сессий, выбирая действия согласно предсказаниям сети.
- **Формирование выборки**: отбираются \(M\) лучших сессий (элитные).
- **Оптимизация**: сеть обучается максимизировать правдоподобие действий из элитных сессий (как в классификаторе):

\[
\pi = \arg\max_\pi \sum_{(s_i, a_i) \in \text{Elite}} \log \pi(a_i|s_i)
\]

Обновление весов:

\[
W_{i+1} = W_i + \alpha \nabla \left[ \sum_{(s_i, a_i) \in \text{Elite}} \log \pi(a_i|s_i) \right]
\]

- **Непрерывное пространство действий**: если действия непрерывны, стратегию часто моделируют как нормальное (гауссово) распределение:

\[
\pi_W(a|s) = \mathcal{N}(\mu(s), \sigma^2)
\]

---

## Неделя №2

### 2.1. V(s), Q(s, a) – что это такое?

**\(V(s)\) – функция ценности состояния (state-value function)** – математическое ожидание суммарного дисконтированного дохода, который агент может получить, начиная с состояния \(s\) и далее следуя своей политике \(\pi\):

\[
V_\pi(s) \triangleq \mathbb{E}_\pi[G_t \mid S_t = s]
\]

где \(G_t\) – дисконтированная сумма наград с момента \(t\):

\[
G_t \triangleq R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad 0 \le \gamma < 1
\]

\(\gamma\) – коэффициент дисконтирования. Он заставляет агента ценить текущую награду выше будущей, а также обеспечивает конечность суммы в непрерывных процессах.

Рекуррентная форма:

\[
G_t = R_t + \gamma G_{t+1}
\]

Уравнение Беллмана для \(V_\pi\):

\[
V_\pi(s) = \sum_a \pi(a|s) \sum_{r,s'} p(r,s'|s,a) \big[ r + \gamma V_\pi(s') \big]
\]

Интуитивно, \(V(s)\) показывает, насколько выгодно для агента находиться в состоянии \(s\), если он продолжает действовать согласно \(\pi\).

**\(Q(s, a)\) – функция ценности действия (action-value function)** – ожидаемое вознаграждение, если агент в состоянии \(s\) сначала выполнит действие \(a\), а затем будет следовать политике \(\pi\):

\[
Q_\pi(s, a) \triangleq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
\]

Уравнение Беллмана для \(Q\):

\[
Q_\pi(s, a) = \sum_{r,s'} p(r,s'|s,a) \big[ r + \gamma V_\pi(s') \big]
\]

Функции \(V\) и \(Q\) используются для определения оптимальной политики \(\pi^*\), которая обеспечивает максимальные значения во всех состояниях. Политика \(\pi\) лучше, чем \(\pi'\), если \(V_\pi(s) \ge V_{\pi'}(s)\) для всех \(s\).

---

### 2.2. Метод Policy Iteration

**Policy Iteration** – алгоритм поиска оптимальной стратегии в Марковских процессах принятия решений (MDP). Он циклически чередует два этапа: **оценку текущей стратегии** и **её улучшение**.

Алгоритм основан на обобщённой итерации стратегий (Generalized Policy Iteration).

#### Этап 1. Инициализация
Начинаем с произвольной стратегии \(\pi\) (например, случайный выбор действий) и произвольной функции ценности \(V(s) = 0\) для всех \(s \in S\).

#### Этап 2. Оценка стратегии (Policy Evaluation)
Фиксируем стратегию \(\pi\) и итеративно обновляем \(V(s)\) по уравнению Беллмана до сходимости:

\[
V(s) = \sum_{s', r} p(s', r \mid s, \pi(s)) \big[ r + \gamma V(s') \big]
\]

#### Этап 3. Улучшение стратегии (Policy Improvement)
Зная точные \(V(s)\) для текущей стратегии, делаем её «жадной» относительно этих ценностей. Для каждого состояния выбираем действие, максимизирующее ожидаемую выгоду:

\[
\pi(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma V(s') \big]
\]

Если новая стратегия совпадает со старой во всех состояниях, алгоритм завершается (найдена оптимальная стратегия). Иначе возвращаемся к этапу 2 с новой стратегией.

> **Сравнение с Value Iteration**  
> Policy Iteration требует полной оценки стратегии на каждом шаге (много итераций внутри), но внешних итераций нужно меньше. Value Iteration делает только одну итерацию оценки на каждом шаге, но требует больше внешних итераций.

---

## Неделя №3

### 3.1. Q-learning, SARSA

В реальных задачах мы редко знаем полную динамику среды, поэтому переходим к **model-free** методам.

**Q-learning** и **SARSA** – алгоритмы обучения с подкреплением из семейства **Temporal Difference (TD)**. Они оба обучают функцию \(Q(s, a)\) без знания модели среды, но различаются подходом к обновлению.

#### Q-learning (off-policy)

Оценивает ценность текущего шага, предполагая, что на следующем шаге агент выберет **абсолютно лучшее** (жадное) действие, даже если реально он будет выбирать по другой стратегии (например, для исследования).

Формула обновления:

\[
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \big( r + \gamma \max_{a'} Q(s', a') \big)
\]

где \(\alpha\) – скорость обучения, \(\gamma\) – коэффициент дисконтирования, \(r + \gamma \max_{a'} Q(s', a')\) – целевое TD-значение.

#### SARSA (on-policy)

Оценивает и улучшает ту же стратегию, по которой агент реально действует. При обновлении учитывается **следующее действие \(a'\)**, которое агент уже выбрал согласно текущей политике.

Формула обновления:

\[
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \big( r + \gamma Q(s', a') \big)
\]

Здесь \(a'\) – действие, выбранное агентом в состоянии \(s'\) (а не максимальное).

---

### 3.2. Exploration vs Exploitation

**Дилемма** – поиск баланса между:

- **Exploitation (использование)** – выбор действий, которые кажутся наиболее выгодными по текущим оценкам, чтобы максимизировать немедленный выигрыш.
- **Exploration (исследование)** – выбор неоптимальных действий для сбора новой информации, поиска глобального оптимума и максимизации долгосрочной награды.

Если агент только эксплуатирует, он рискует застрять в локальном оптимуме. Если только исследует – он не накопит высокую награду из-за постоянных ошибок.

#### Основные методы решения

- **\(\varepsilon\)-greedy**  
  С вероятностью \(\varepsilon\) выбирается случайное действие (exploration), с вероятностью \(1-\varepsilon\) – лучшее согласно текущим оценкам (exploitation). На практике \(\varepsilon\) уменьшают со временем.

- **Softmax**  
  Действия выбираются пропорционально их ценности:

  \[
  \pi(a|s) = \frac{\exp\left( Q(s,a) / T \right)}{\sum_{a_i} \exp\left( Q(s,a_i) / T \right)}
  \]

  где \(T\) – температура. При высокой \(T\) выбор почти равномерный, при низкой – агент склоняется к лучшему действию.

---

## Неделя №4

### 4.1. DQN

**Deep Q-Network (DQN)** – алгоритм, объединяющий Q-learning с глубокими нейронными сетями. Вместо таблицы \(Q(s,a)\) используется нейросеть-аппроксиматор, которая на вход принимает состояние \(s\) и выдаёт Q-значения для всех возможных действий.

Преимущество: один проход сети даёт оценки для всех действий сразу, что эффективно для больших пространств состояний (например, изображения). Архитектура часто содержит свёрточные слои для обработки визуальных данных, затем полносвязные слои (без активации на выходе).

#### Проблемы обучения и их решения

Обучение глубоких сетей в RL нестабильно, поэтому DQN использует две ключевые техники:

- **Experience Replay** – сохранение прошлых переходов \((s, a, r, s')\) в буфер и обучение на случайных мини-батчах, что разрушает корреляцию последовательных данных.
- **Target Network** – отдельная сеть с «замороженными» весами для вычисления целевого значения, что стабилизирует обучение.

Функция потерь (MSE):

\[
L(\Theta) = \mathbb{E}_{(s,a,r,s') \sim \text{Buffer}} \left[ \big( Q(s,a,\Theta) - (r + \gamma \max_{a'} Q(s', a', \Theta^-)) \big)^2 \right]
\]

где \(\Theta\) – параметры основной сети, \(\Theta^-\) – параметры целевой сети.

---

### 4.2. Experience Replay

**Experience Replay (воспроизведение опыта)** – техника, позволяющая эффективно использовать накопленные данные.

- Агент сохраняет кортежи \((s, a, r, s')\) в буфер (Replay Buffer), который может хранить до \(10^6\) последних переходов.
- Во время обучения из буфера случайно выбираются мини-батчи для обновления сети.

#### Решаемые проблемы

- **Нарушение независимости данных** – последовательные шаги коррелированы; случайная выборка перемешивает историю, делая данные более независимыми.
- **Борьба с забыванием** – буфер позволяет модели «вспоминать» старый опыт.
- **Эффективность данных** – один и тот же опыт может использоваться многократно.

---

### 4.3. Target Network

**Target Network (целевая сеть)** – вспомогательная сеть для стабилизации обучения DQN.

В классическом Q-learning целевое значение вычисляется как \(r + \gamma \max_{a'} Q(s', a')\). Если использовать одну и ту же сеть и для предсказания, и для цели, то при обновлении весов цель тоже сдвигается, что ведёт к осцилляциям и расходимости.

**Решение**: использовать отдельную сеть с параметрами \(\Theta^-\), которые обновляются реже или плавно.

Функция потерь:

\[
L(\Theta) = \mathbb{E}_{(s,a,r,s')} \left[ \big( Q(s,a,\Theta) - (r + \gamma \max_{a'} Q(s', a', \Theta^-)) \big)^2 \right]
\]

#### Способы обновления \(\Theta^-\)

- **Hard Target Network** – полное копирование весов из основной сети каждые \(n\) шагов.
- **Soft Target Network** – плавное обновление с помощью экспоненциального скользящего среднего:

  \[
  \Theta^- \leftarrow (1 - \alpha) \Theta^- + \alpha \Theta, \quad \alpha \ll 1
  \]

  Это обеспечивает максимальную стабильность целей.

---

### 4.4. Double DQN

**Double DQN (DDQN)** – модификация DQN, устраняющая систематическое завышение оценок действий.

В стандартном DQN целевое значение использует оператор максимума:

\[
\text{Target}_{\text{DQN}} = r + \gamma \max_{a'} Q(s', a', \Theta^-)
\]

Из-за шума в предсказаниях сети максимум часто оказывается завышенным (математическое ожидание максимума ≥ максимум ожиданий). Это приводит к нестабильной стратегии.

**Идея DDQN**: разделить выбор действия и его оценку между двумя разными сетями. Поскольку в DQN уже есть основная (\(\Theta\)) и целевая (\(\Theta^-\)) сети, используем их так:

- **Выбор действия** – выполняется основной сетью: \(a^* = \arg\max_{a'} Q(s', a', \Theta)\).
- **Оценка действия** – выполняется целевой сетью: \(Q(s', a^*, \Theta^-)\).

Целевое значение DDQN:

\[
r + \gamma \, Q(s', \arg\max_{a'} Q(s', a', \Theta), \Theta^-)
\]

Это снижает переоценку, так как выбор и оценка используют разные наборы весов.

---

## Неделя №6

### 5.1. Policy Gradient

**Policy Gradient** – семейство методов, оптимизирующих параметры стратегии напрямую (в отличие от value-based методов, где сначала оценивают полезность действий).

#### Основная идея

- Прямая максимизация ожидаемой награды через настройку параметров \(\theta\) стратегии \(\pi_\theta(a|s)\).
- Выбор действия прост, особенно в сложных средах, где оценить ценность каждого действия трудно.
- Встроенное исследование среды – стохастическая стратегия (\(a \sim \pi_\theta(a|s)\)) автоматически обеспечивает exploration.

#### Целевая функция и градиент

Целевая функция – ожидаемая дисконтированная награда за сессию:

\[
J(\theta) = \mathbb{E}_{s \sim p, a \sim \pi_\theta} [ G(s, a) ]
\]

Проблема: параметры \(\theta\) скрыты внутри распределения, поэтому нельзя просто продифференцировать \(J\) по сэмплированным траекториям.

Используется **трюк с логарифмической производной**:

\[
\nabla \log \pi(z) = \frac{1}{\pi(z)} \nabla \pi(z) \quad \Rightarrow \quad \nabla \pi(z) = \pi(z) \nabla \log \pi(z)
\]

Тогда градиент \(J\):

\[
\nabla J = \mathbb{E} \big[ \nabla \log \pi_\theta(a|s) \, Q(s, a) \big]
\]

---

### 5.2. REINFORCE

**REINFORCE** – базовый алгоритм Policy Gradient, on-policy.

#### Алгоритм

1. Инициализация весов \(\theta\) случайно.
2. Генерация \(N\) сессий с использованием текущей стратегии \(\pi_\theta\).
3. Оценка градиента:

   \[
   \nabla J \approx \frac{1}{N} \sum_{i=0}^{N} \sum_{s,a} \nabla \log \pi_\theta(a|s) \, Q(s, a)
   \]

4. Обновление параметров градиентным подъёмом:

   \[
   \theta \leftarrow \theta + \alpha \nabla J
   \]

#### Уменьшение дисперсии (базовая линия)

Классический REINFORCE страдает от высокой дисперсии. Для стабилизации используют **базовую линию** \(b(s)\) (часто \(V(s)\)):

\[
\nabla J \approx \frac{1}{N} \sum_{i=0}^{N} \sum_{s,a} \nabla \log \pi_\theta(a|s) \big( Q(s, a) - b(s) \big)
\]

Вычитание \(b(s)\) не меняет математическое ожидание градиента, но значительно снижает дисперсию.

---

### 5.3. Actor-critic

**Actor-Critic** – гибридный подход, объединяющий методы на основе стратегии (policy-based) и на основе ценности (value-based).

В REINFORCE оценка успешности действия происходит только в конце сессии, что приводит к высокой дисперсии и медленному обучению. Actor-Critic решает эту проблему, разделяя агента на две части:

- **Actor** – модель стратегии \(\pi_\theta(a|s)\), выбирает действия.
- **Critic** – модель ценности \(V(s)\), оценивает, насколько удачным было выбранное действие, и помогает Actor учиться быстрее.

#### Advantage Actor-Critic (A2C)

Critic вычисляет **функцию преимущества** \(A(s, a)\), показывающую, насколько конкретное действие лучше среднего для этого состояния:

\[
A(s, a) = r + \gamma V(s') - V(s)
\]

где \(r\) – награда, \(V(s')\) – оценка ценности следующего состояния, \(V(s)\) – оценка текущего.

#### Процесс обучения

1. Actor по состоянию \(s\) выбирает действие \(a \sim \pi_\theta(a|s)\).
2. Среда возвращает награду \(r\) и следующее состояние \(s'\).
3. Critic вычисляет ошибку временного различия (TD-ошибку): \(\delta = r + \gamma V(s') - V(s)\).
4. Если \(\delta > 0\), действие лучше ожидаемого; если \(\delta < 0\) – хуже.
5. Critic минимизирует MSE предсказания \(V(s)\).
6. Actor корректирует параметры стратегии: хорошие действия поощряются, плохие – штрафуются.

Этот подход обеспечивает более стабильное и быстрое обучение по сравнению с REINFORCE.

---

## Неделя №7

### 6.1. Применение RNN для генерации последовательностей

В задачах генерации последовательностей наиболее распространенным подходом является архитектура **Encoder-Decoder** (кодировщик-декодировщик):

- **Encoder (Кодировщик)** – считывает входные данные, которые могут представлять собой последовательность произвольной длины или другие типы данных. Он сжимает эту информацию в единое векторное представление (скрытое состояние).
- **Decoder (Декодировщик)** – пошагово генерирует выходную последовательность, принимая скрытый вектор и ранее сгенерированные токены, пока не будет выдан специальный символ окончания.

#### Основные сферы применения

- **Машинный перевод** – сеть считывает предложение на одном языке и генерирует эквивалентное по смыслу предложение на другом языке.
- **Генерация описаний к изображениям** – на вход подаются признаки изображения, извлеченные сверточной нейросетью (CNN), а декодировщик RNN генерирует связный текст, описывающий происходящее.
- **Преобразование графем в фонемы** – чтение буквенной записи слова и генерация его фонетической транскрипции.
- **Диалоговые системы и чат-боты** – анализ реплики пользователя и пошаговая генерация текстового ответа системы.
- **Другие задачи** – перевод изображений математических формул в код LaTeX, преобразование исходного кода в строку документации и транскрипция аудио.

Модели обучаются с помощью **обучения с учителем (Supervised Learning)** на больших наборах данных, состоящих из пар «вход-эталон». Цель обучения – максимизировать логарифм вероятности эталонного ответа при заданном входе.

Основной недостаток этого подхода – **сдвиг распределения (distribution shift)**. Во время обучения модель видит идеальные предыдущие токены из данных, но при генерации она опирается на свои собственные (возможно, ошибочные) предсказания, что приводит к накоплению ошибок.

---

### 6.2. Применение RL к дообучению моделей генерации последовательности

Обучение с подкреплением (Reinforcement Learning, RL) используется на этапе дообучения, чтобы преодолеть ограничения обучения с учителем и улучшить качество генерации. В отличие от обучения с учителем, RL позволяет:

- Обучать модель на её собственных результатах (устраняя сдвиг распределения).
- Напрямую оптимизировать недифференцируемые метрики (например, BLEU, ROUGE или человеческие предпочтения).

#### Основные подходы

- **Self-Critical Sequence Training (SCST)**  
  Идея заключается в использовании собственного веса модели в режиме тестирования (greedy inference) в качестве базовой линии (baseline). Модель обучается на сэмплированных последовательностях, получая вознаграждение, если они оказываются лучше, чем результат «жадной» генерации.

- **RLHF (Reinforcement Learning from Human Feedback)**  
  Этот метод лежит в основе таких моделей, как ChatGPT. Он включает три этапа:
  1. **Предварительное обучение с учителем (SFT)** – базовая модель обучается на больших корпусах текста.
  2. **Обучение модели вознаграждения (Reward Model)** – на основе человеческих рейтингов (предпочтений) обучается модель, которая предсказывает оценку качества ответа.
  3. **Финальная оптимизация стратегии (policy)** – с помощью алгоритма **PPO (Proximal Policy Optimization)** дообучается основная модель, максимизируя оценку от модели вознаграждения.

- **DPO (Direct Preference Optimization)**  
  Более простой и стабильный метод, который позволяет оптимизировать модель под предпочтения человека напрямую, без необходимости обучения отдельной модели вознаграждения и использования сложных RL-алгоритмов. DPO переводит задачу оптимизации RLHF в задачу классификации на парах предпочтений.

---












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


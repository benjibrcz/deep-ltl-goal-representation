# Full Network Architecture

This diagram illustrates the complete neural network architecture, showing how environment states are processed through observation encoding, Büchi tracking, sequence extraction, value evaluation, and finally action selection in the full network.

```mermaid
graph TD
    %% Per-timestep data flow
    ENV[Environment<br/>state&nbsp;s_t] --> OE[Observation<br/>Encoder]
    OE --> OB[obs_emb]

    ENV --> AP[AP&nbsp;labeling<br/>L(s_t)]
    AP --> BT[Büchi&nbsp;Tracker<br/>q_t]
    BT --> EX[Reach-Avoid<br/>Seq Extractor]
    EX --> EN[Sequence<br/>Encoder]
    EN --> SEQ[seq_emb]

    OB --> VAL[Value Module]
    SEQ --> VAL
    VAL --> SC[Value&nbsp;scores]

    SC --> SEL[Sequence&nbsp;Selector<br/>best&nbsp;σ*]

    OB --> ACT[Actor Module<br/>(Policy)]
    SEL --> ACT
    ACT --> ACTN[Action&nbsp;a_t]
    ACTN --> ENV
```

## Network Components

- **Environment**: Provides the current state `s_t`
- **Observation Encoder**: Neural network that processes environment state into observation embeddings
- **AP Labeling**: Extracts atomic propositions from the current state
- **Büchi Tracker**: Maintains the current Büchi automaton state `q_t`
- **Reach-Avoid Seq Extractor**: Extracts relevant sequences based on current Büchi state
- **Sequence Encoder**: Neural network that encodes the extracted sequences into embeddings
- **Value Module**: Neural network that evaluates different sequences and produces value scores
- **Sequence Selector**: Chooses the best sequence `σ*` based on value scores
- **Actor Module**: The policy network that produces actions based on observation embeddings and selected sequence
- **Action**: The final action `a_t` that is executed in the environment 
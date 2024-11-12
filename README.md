## Demo

- Start weaviate docker container

### Backstory - New coach for a baseball team

Oh no, the old coach has moved out of town! You have taken over the team.

The players know their positions in the field, but the coach didn't leave behind a batting order.

Luckily we know the players on the team and their hitting stats from the previous season.

Let's calculate the OPS for each player and use the top 9 batters in our lineup!

```bash
poetry run make-lineup
```

#### What did we learn?

`Gradient Boosting` is a machine learning technique that works well for make predictions from tabular data. 

> [!TIP]
> Collect data that includes the values you would like to predict.

> [!TIP]
> Be prepared to clean your data and normalize your data if necessary.

### Learn To Call Pitches

I haven't coached before, let's simulate a game to get some practice. In particular, let's focus on the pitches the pitcher should throw given the game situation.

```bash
poetry run learn-to-pitch
```

#### What did we learn?

Use `Deep Reinforcement Learning` to create models that learn which action to take based on a reward function. 

> [!TIP]
> Too much training can overfit your model. Insufficient training can underfit your model which means that the predictions it makes will not be very good.
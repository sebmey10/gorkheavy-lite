"""
So, the goal is to create an AI workflow that sends multiple agents to answer one query.
Each answer from each agent is then fed to a "judge" AI that, in turn, chooses which agent came up with the best answer.
If you look at "gorkheavy-lite.py" you can see the theoretical function of what the workflow will kind of look like.

In terms of setting up the workflow in a k3s cluster, separating each agent llm into it's own container seems
to be the best route of action. Every separate aspect of each function, actually, will be separated into it's 
own container. Listed as follows:

1. Each llm needs to be containerized. (Done)
2. The coordination of the workflow needs to be containerized. AKA the script defining how all running parts coincide
into one working function.

That list turned out to be shorter than I imagined. However, 2 is (and you can't change my mind) a large number.

Our job in "final_command_script" is to make a script, like "gorkheavy-lite.py", but it is going to coordinate
all moving parts in a way that our cluster will respond well to.

Take peeks at what I'm up to at your own discretion, can't promise what I'm up to will work.

"""

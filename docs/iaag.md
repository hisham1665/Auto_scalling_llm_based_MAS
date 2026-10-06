# IAAG

Initial Automatic Agent Generation starts from predefined seed agents, records the task, and asks the generator to identify missing expertise before any agent response. Valid generated specifications pass through `AgentFactory` and `AgentRegistry`. The initial generation loop stops when the manager returns `create_agent: false` or a hard guard is reached. The first conversation turn is only selected after this process, which is the IAAG validation invariant.

The static baseline bypasses this loop and uses the scenario’s complete user-defined roster. IAAG uses only seed agents initially; automatically generated agents are recorded with `created_dynamically=true` and a reason.

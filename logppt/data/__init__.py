from .base import load_loghub_dataset

DATASET_FORMAT = {
    "HDFS": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "Hadoop": "<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>",
    "Spark": "<Date> <Time> <Level> <Component>: <Content>",
    "Zookeeper": "<Date> <Time> - <Level>  \\[<Node>:<Component>@<Id>\\] - <Content>",
    "BGL": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
    "HPC": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
    "Thunderbird": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>",
    "Windows": "<Date> <Time>, <Level>                  <Component>    <Content>",
    "Linux": "<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>",
    "Android": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
    "HealthApp": "<Time>\\|<Component>\\|<Pid>\\|<Content>",
    "Apache": "\\[<Time>\\] \\[<Level>\\] <Content>",
    "Proxifier": "\\[<Time>\\] <Program> - <Content>",
    "OpenSSH": "<Date> <Day> <Time> <Component> sshd\\[<Pid>\\]: <Content>",
    "OpenStack": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \\[<ADDR>\\] <Content>",
    "Mac": "<Month>  <Date> <Time> <User> <Component>\\[<PID>\\]( \\(<Address>\\))?: <Content>"
}
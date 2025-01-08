# gym-hpa

The implemented gym-hpa is a custom [OpenAi Gym](https://gym.openai.com/) 
environment for the training of Reinforcement Learning (RL) agents for auto-scaling research 
in the Kubernetes (K8s) platform. 


## How does it work?

Two environments exist based on the [Redis Cluster](https://github.com/bitnami/charts/tree/master/bitnami/redis-cluster) and [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) applications. 

Both RL environments have been designed: actions, observations, reward function. 

Please check the [run.py](policies/run/run.py) file to understand how to run the framework. 

To run in the real cluster mode, you should add the token to your cluster [here](gym_hpa/envs/deployment.py)

### Running

To run the code, go to the folder `policies/run` and run:

```bash
python run.py
```

Additional arguments can be passed while running run.py. Please check here [run.py](policies/run/run.py). 

## Team

* [Jose Santos](https://scholar.google.com/citations?hl=en&user=57EIYWcAAAAJ)

* [Tim Wauters](https://scholar.google.com/citations?hl=en&user=Kvxp9iYAAAAJ)

* [Bruno Volckaert](https://scholar.google.com/citations?hl=en&user=NIILGOMAAAAJ)

* [Filip de Turck](https://scholar.google.com/citations?hl=en&user=-HXXnmEAAAAJ)

## Contact

If you want to contribute, please contact:

Lead developer: [Jose Santos](https://github.com/jpedro1992/)

For questions or support, please use GitHub's issue system.

## License

Copyright (c) 2020 Ghent University and IMEC vzw.

Address: IDLab, Ghent University, iGent Toren, Technologiepark-Zwijnaarde 126 B-9052 Gent, Belgium



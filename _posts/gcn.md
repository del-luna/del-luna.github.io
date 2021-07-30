---

---



## Contribution

- 제안한 방법이 Spectral graph convolution의 1차 근사임을 보임.
- 제안한 방법이 빠르고 확장 가능한 Semi-supervised에서 어떻게 사용될 수 있는지를 보임.



선행 연구에서는 $\mathcal L=\mathcal{L}_0 + \mathcal{L}_{reg}$ 과 같은 식의 로스를 사용하는데 뒤에 있는 term인 $\mathcal{L}_{reg}$같은 경우 Graph Laplacian Regulatizaion이다. 이는 연결된 노드가 비슷한 representation을 가지도록 하는 정규화 텀으로 연결된 노드가 도일한 레이블을 공유할 가능성이 있다는 가정에 의존한다. 
이로 인해 모델의 표현력이 단순한 유사도 이외에는 제한되기 때문에 본 논문에서는 인접행렬을 인풋으로 하여 노드 연결에 대한 정보의 제한 문제를 해결함.



$$H^{(l+1)} = \sigma(\tilde D^{-\frac12}\tilde A \tilde D^{-\frac12}H^{(l)}W{(l)})$$

- $H^{(l)}$ : $l$번째 레이어의 히든 스테이트, $H^0 = X$ 는 Graph node init feature
- $\tilde A$ : $A+I_N$ 
- $\tilde D$ : 


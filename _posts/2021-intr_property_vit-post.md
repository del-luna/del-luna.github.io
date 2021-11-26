---

---



Properties of Vit

- Tf는 domain shift, sever occlusions, perturbations에 대해 robust함.
- 위와 같은 robust는 texture biased로 인한 것이 아니라 Vit가 CNN에 비해 local texture에 대해 훨씬 덜 biased 되어있음을 보여준다. shape-based feature를 인코딩 하도록 학습된 경우 human-level의 shape recognition을 보여줌.
- pixel-level supervision 없이 정확한 semantic segmentation결과를 얻을 수 있음.
- 앙상블도 잘 됨.

결론 : 위의 프로퍼티들이 가능한 이유는 self-attn을 통해 유연한 receptive field가 가능하기 때문이다.


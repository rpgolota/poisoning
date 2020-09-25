# Dataset Poisoning
---

## Progress

- [ ] Full algorithm implemented
  - [ ] Feature selection algorithms
    - In sklearn.linear_model
    - [ ] Lasso
    - [ ] Ridge
    - [ ] Elastic Net
  - [ ] Attack stratagy
    - Referring to equation 3 
  - [ ] Attack stratagy gradient
    - Referring to equation 4

### Equations

$EQ2:~~~ \min\limits_{\boldsymbol{w}, b} \mathcal{L} = \frac{1}{n}\sum^n\limits_{i=1}\ell(y_i, f(\boldsymbol{x}_i)) + \lambda\Omega(\boldsymbol{w}) $

$EQ3:~~~ \max\limits_{\boldsymbol{x}_c} \mathcal{W} = \frac{1}{m}\sum^m\limits_{j=1}\ell(\hat{y_j}, f(\hat{\boldsymbol{x}_j})) + \lambda\Omega(\boldsymbol{w}) $

$EQ4:~~~ \frac{\partial{\mathcal{W}}}{\partial{{\boldsymbol{x}_c}}} = \frac{1}{m}\sum^m\limits_{j=1}(f(\hat{\boldsymbol{x}_j})-\hat{y_j})\bigg(\hat{\boldsymbol{x}}^\intercal_j\frac{\partial{\boldsymbol{w}}}{\partial{\boldsymbol{x}_c}}+\frac{\partial{b}}{\partial{\boldsymbol{x}_c}}\bigg) + \lambda\boldsymbol{r}\frac{\partial{\boldsymbol{w}}}{\partial{\boldsymbol{x}_c}}$
# Conditional independence tests

Two tests are bundled. Both implement the [`CITest`](#cdans.ci_tests.CITest)
protocol, so you can also write your own and pass it as `ci_test=` to
[`CDANs`](cdans.md).

## Picking a test

| Test       | Cost     | Assumes      | Use when                                       |
| ---------- | -------- | ------------ | ---------------------------------------------- |
| `FisherZ`  | O(n)     | linear, Gaussian | data is approximately linear-Gaussian          |
| `KCITest`  | O(n³)    | none         | dependencies may be nonlinear; surrogate-dependence check |

Pick by name (`ci_test="fisherz"` or `"kci"`) or by passing an instance for
custom configuration.

::: cdans.ci_tests.CITest

::: cdans.ci_tests.FisherZ

::: cdans.ci_tests.kci.KCITest

::: cdans.ci_tests.get_ci_test

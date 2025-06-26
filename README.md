# Fraud-detector-mlops

```
docker run -p 5000:5000 mlflow-server
docker run -p 5000:5000 mlflow/mlflow
```

# Dataset Sintético de Risco de Empréstimo

Este dataset foi gerado sinteticamente para simular um cenário de classificação de risco de empréstimo. A variável alvo (`risco_emprestimo`) possui três categorias: `baixo`, `moderado` e `alto`.

As variáveis explicativas são:

`renda_mensal` (float): renda mensal do solicitante em reais, gerada por uma distribuição normal com média 5000 e desvio padrão 1500, truncando valores abaixo de 800. Exemplo de geração: `renda_mensal = np.random.normal(loc=5000, scale=1500, size=n_samples).clip(800, None)`

`score_credito` (int): score de crédito do solicitante, entre 300 e 1000, gerado por valores inteiros aleatórios uniformes. Exemplo: `score_credito = np.random.randint(300, 1001, size=n_samples)`

`dividas_atuais` (float): total de dívidas atuais do cliente em reais, gerado por distribuição normal com média 10000 e desvio padrão 5000, truncando valores negativos para zero. Exemplo: `dividas_atuais = np.random.normal(loc=10000, scale=5000, size=n_samples).clip(0, None)`

`numero_atrasos` (int): número de pagamentos em atraso nos últimos 12 meses, gerado como valores inteiros entre 0 e 10. Exemplo: `numero_atrasos = np.random.randint(0, 11, size=n_samples)`

A variável alvo `risco_emprestimo` é calculada por meio de uma regra ponderada com base nas variáveis acima, onde o risco aumenta com o aumento da dívida relativa à renda, queda no score de crédito e aumento nos atrasos. A fórmula usada é:

`risco_score = (divida / renda) * 0.4 + (10 - score / 1000 * 10) * 0.3 + atrasos * 0.3`

Com a classificação:

- Se `risco_score < 4`, risco = "baixo"
- Se `risco_score < 7`, risco = "moderado"
- Caso contrário, risco = "alto"

Essa regra reflete que o risco é maior quando a dívida é alta em relação à renda, o score de crédito é baixo e há muitos atrasos em pagamentos anteriores.

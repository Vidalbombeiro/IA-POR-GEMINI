# Exemplo de Saída do Programa

Quando você executar `python ia_iniciante.py`, verá uma saída como esta:

```
============================================================
 🤖 BEM-VINDO À SUA PRIMEIRA INTELIGÊNCIA ARTIFICIAL! 🤖
============================================================

Vamos ensinar uma IA a diferenciar números PARES de ÍMPARES!

🎓 Iniciando treinamento com 30 exemplos...
   Taxa de aprendizado: 0.1
   Épocas: 100

   Época 20/100 - Precisão: 100.0%
   Época 40/100 - Precisão: 100.0%
   Época 60/100 - Precisão: 100.0%
   Época 80/100 - Precisão: 100.0%
   Época 100/100 - Precisão: 100.0%

✅ Treinamento concluído!
   Peso final: -0.4551
   Bias final: 0.1270

🧪 Testando a IA com números novos...

   ✓ 32 → Previsão: PAR (Correto!)
   ✓ 31 → Previsão: ÍMPAR (Correto!)
   ✓ 75 → Previsão: ÍMPAR (Correto!)
   ✓ 62 → Previsão: PAR (Correto!)
   ✓ 43 → Previsão: ÍMPAR (Correto!)
   ✓ 61 → Previsão: ÍMPAR (Correto!)
   ✓ 38 → Previsão: PAR (Correto!)
   ✓ 33 → Previsão: ÍMPAR (Correto!)
   ✓ 74 → Previsão: PAR (Correto!)
   ✓ 42 → Previsão: PAR (Correto!)

📊 Resultado: 10/10 acertos (100%)

🎮 Modo interativo - Teste você mesmo!
   Digite um número para a IA classificar (ou 'sair' para encerrar)

   Digite um número: 42
   🤖 A IA diz: PAR - ✓ Correto!

   Digite um número: 13
   🤖 A IA diz: ÍMPAR - ✓ Correto!

   Digite um número: 100
   🤖 A IA diz: PAR - ✓ Correto!

   Digite um número: sair

👋 Até logo!
```

## O que está acontecendo?

1. **Treinamento**: A IA recebe 30 exemplos de números com suas classificações
2. **Aprendizado**: Em 100 épocas, a IA ajusta seus pesos internos
3. **Teste Automático**: A IA é testada com 10 números novos que nunca viu
4. **Modo Interativo**: Você pode testar com seus próprios números!

A precisão de 100% mostra que a IA aprendeu perfeitamente o padrão!

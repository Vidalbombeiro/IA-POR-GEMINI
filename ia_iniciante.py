#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IA Iniciante - Primeira Inteligência Artificial
================================================
Este programa implementa um Perceptron simples que aprende a 
diferenciar números pares de ímpares.

É o "Hello World" do Machine Learning!
"""

import random


class PerceptronSimples:
    """
    Perceptron - O modelo mais simples de rede neural.
    
    Um perceptron aprende a classificar dados ajustando seus pesos
    através de exemplos de treinamento.
    """
    
    def __init__(self, taxa_aprendizado=0.1):
        """
        Inicializa o perceptron.
        
        Args:
            taxa_aprendizado: Velocidade com que o perceptron aprende (0.0 a 1.0)
        """
        self.taxa_aprendizado = taxa_aprendizado
        self.peso = random.uniform(-1, 1)  # Peso inicial aleatório
        self.bias = random.uniform(-1, 1)  # Bias inicial aleatório
    
    def extrair_caracteristica(self, numero):
        """
        Extrai uma característica do número que ajuda a identificar se é par ou ímpar.
        
        Args:
            numero: Número de entrada
            
        Returns:
            Resto da divisão por 2 (0 para par, 1 para ímpar)
        """
        return numero % 2
        
    def ativacao(self, x):
        """
        Função de ativação - decide se o neurônio "dispara" ou não.
        
        Args:
            x: Valor de entrada
            
        Returns:
            1 se x >= 0, caso contrário -1
        """
        return 1 if x >= 0 else -1
    
    def prever(self, entrada):
        """
        Faz uma previsão para uma entrada.
        
        Args:
            entrada: Número para classificar
            
        Returns:
            1 para par, -1 para ímpar
        """
        # Extrai a característica (resto da divisão por 2)
        caracteristica = self.extrair_caracteristica(entrada)
        # Calcula a soma ponderada
        soma = caracteristica * self.peso + self.bias
        # Aplica a função de ativação
        return self.ativacao(soma)
    
    def treinar(self, entradas, rotulos, epocas=100):
        """
        Treina o perceptron com exemplos.
        
        Args:
            entradas: Lista de números para treinar
            rotulos: Lista de classificações corretas (1=par, -1=ímpar)
            epocas: Número de vezes que passamos por todos os exemplos
        """
        total_exemplos = len(entradas)
        print(f"🎓 Iniciando treinamento com {total_exemplos} exemplos...")
        print(f"   Taxa de aprendizado: {self.taxa_aprendizado}")
        print(f"   Épocas: {epocas}\n")
        
        for epoca in range(epocas):
            erros = 0
            
            # Para cada exemplo de treinamento
            for entrada, rotulo_correto in zip(entradas, rotulos):
                # Extrai característica
                caracteristica = self.extrair_caracteristica(entrada)
                # Calcula soma ponderada e faz previsão
                soma = caracteristica * self.peso + self.bias
                previsao = self.ativacao(soma)
                
                # Calcula o erro
                erro = rotulo_correto - previsao
                
                if erro != 0:
                    erros += 1
                    # Ajusta os pesos (aqui está o aprendizado!)
                    self.peso += self.taxa_aprendizado * erro * caracteristica
                    self.bias += self.taxa_aprendizado * erro
            
            # Mostra progresso a cada 20 épocas
            if (epoca + 1) % 20 == 0:
                precisao = ((total_exemplos - erros) / total_exemplos) * 100
                print(f"   Época {epoca + 1}/{epocas} - Precisão: {precisao:.1f}%")
        
        print(f"\n✅ Treinamento concluído!")
        print(f"   Peso final: {self.peso:.4f}")
        print(f"   Bias final: {self.bias:.4f}\n")


def gerar_dados_treinamento(quantidade=20):
    """
    Gera dados de treinamento (números e suas classificações).
    
    Args:
        quantidade: Quantos exemplos gerar
        
    Returns:
        Tupla (entradas, rótulos)
    """
    entradas = []
    rotulos = []
    
    for _ in range(quantidade):
        # Gera um número aleatório entre 0 e 100
        numero = random.randint(0, 100)
        entradas.append(numero)
        
        # Classifica: 1 para par, -1 para ímpar
        if numero % 2 == 0:
            rotulos.append(1)  # Par
        else:
            rotulos.append(-1)  # Ímpar
    
    return entradas, rotulos


def verificar_acerto(previsao, numero):
    """
    Verifica se a previsão está correta.
    
    Args:
        previsao: Previsão da IA (1 para par, -1 para ímpar)
        numero: Número que foi classificado
        
    Returns:
        Tupla (acertou, eh_par, previsao_texto, correto_texto)
    """
    eh_par = (numero % 2 == 0)
    previsao_texto = "PAR" if previsao == 1 else "ÍMPAR"
    correto_texto = "PAR" if eh_par else "ÍMPAR"
    acertou = (previsao == 1 and eh_par) or (previsao == -1 and not eh_par)
    
    return acertou, eh_par, previsao_texto, correto_texto


def testar_ia(perceptron, quantidade_testes=10):
    """
    Testa a IA com novos números.
    
    Args:
        perceptron: O perceptron treinado
        quantidade_testes: Quantos testes realizar
    """
    print("🧪 Testando a IA com números novos...\n")
    
    acertos = 0
    
    for _ in range(quantidade_testes):
        # Gera um número aleatório
        numero = random.randint(0, 100)
        
        # Pede para a IA prever
        previsao = perceptron.prever(numero)
        
        # Verifica se acertou usando a função helper
        acertou, _, previsao_texto, correto_texto = verificar_acerto(previsao, numero)
        
        if acertou:
            acertos += 1
            print(f"   ✓ {numero} → Previsão: {previsao_texto} (Correto!)")
        else:
            print(f"   ✗ {numero} → Previsão: {previsao_texto} (Era: {correto_texto})")
    
    precisao = (acertos / quantidade_testes) * 100
    print(f"\n📊 Resultado: {acertos}/{quantidade_testes} acertos ({precisao:.0f}%)\n")


def modo_interativo(perceptron):
    """
    Permite ao usuário testar a IA com seus próprios números.
    
    Args:
        perceptron: O perceptron treinado
    """
    print("🎮 Modo interativo - Teste você mesmo!")
    print("   Digite um número para a IA classificar (ou 'sair' para encerrar)\n")
    
    while True:
        try:
            entrada = input("   Digite um número: ")
            
            if entrada.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Até logo!\n")
                break
            
            numero = int(entrada)
            
            # IA faz a previsão
            previsao = perceptron.prever(numero)
            
            # Verifica se está correto usando a função helper
            acertou, _, previsao_texto, correto_texto = verificar_acerto(previsao, numero)
            
            if acertou:
                print(f"   🤖 A IA diz: {previsao_texto} - ✓ Correto!\n")
            else:
                print(f"   🤖 A IA diz: {previsao_texto} - ✗ Errado! (É {correto_texto})\n")
                
        except ValueError:
            print("   ⚠️  Por favor, digite um número válido!\n")
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!\n")
            break


def main():
    """
    Função principal do programa.
    """
    print("\n" + "="*60)
    print(" 🤖 BEM-VINDO À SUA PRIMEIRA INTELIGÊNCIA ARTIFICIAL! 🤖")
    print("="*60)
    print("\nVamos ensinar uma IA a diferenciar números PARES de ÍMPARES!\n")
    
    # Cria o perceptron
    ia = PerceptronSimples(taxa_aprendizado=0.1)
    
    # Gera dados de treinamento
    entradas, rotulos = gerar_dados_treinamento(quantidade=30)
    
    # Treina a IA
    ia.treinar(entradas, rotulos, epocas=100)
    
    # Testa a IA
    testar_ia(ia, quantidade_testes=10)
    
    # Modo interativo
    modo_interativo(ia)


if __name__ == "__main__":
    main()

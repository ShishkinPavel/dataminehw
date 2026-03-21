# Agent: LLM Bonus HW4

Реализуй LLM-скилл в AL-пайплайне (+1 балл).

## Идея

Claude API анализирует примеры, выбранные стратегией, и объясняет ПОЧЕМУ они информативны. Также может рекомендовать стратегию на основе текущего состояния цикла.

## Реализация

Добавь два метода в `agents/al_agent.py`:

### 1. llm_explain_selection

```python
def llm_explain_selection(
    self,
    selected_texts: list[str],
    strategy: str,
    iteration: int
) -> str:
    """Ask Claude to explain why selected examples are informative.

    Args:
        selected_texts: Texts chosen by the query strategy.
        strategy: Strategy name ('entropy', 'margin', 'random').
        iteration: Current AL iteration number.

    Returns:
        String with LLM explanation.
    """
    try:
        import anthropic
    except ImportError:
        return "anthropic not installed"

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return "ANTHROPIC_API_KEY not set"

    # Показать первые 5 примеров (обрезанных)
    examples = "\n".join(
        f"{i+1}. {t[:200]}..." if len(t) > 200 else f"{i+1}. {t}"
        for i, t in enumerate(selected_texts[:5])
    )

    prompt = f"""You are an Active Learning expert. Analyze these examples selected by the '{strategy}' strategy at iteration {iteration}.

Selected examples:
{examples}

Questions:
1. What makes these examples potentially informative for a sentiment classifier?
2. Do you see patterns (ambiguous sentiment, mixed opinions, sarcasm, unusual vocabulary)?
3. Would a human annotator find these examples easy or hard to label?

Respond in Russian, 3-5 sentences total. Be specific about the examples."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"LLM error: {e}"
```

### 2. llm_recommend_strategy

```python
def llm_recommend_strategy(self, histories: dict[str, list[dict]]) -> str:
    """Ask Claude to recommend the best strategy based on results.

    Args:
        histories: {'entropy': [...], 'random': [...], ...}

    Returns:
        String with recommendation.
    """
    try:
        import anthropic
        import json
    except ImportError:
        return "anthropic not installed"

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return "ANTHROPIC_API_KEY not set"

    summary = json.dumps(histories, indent=2)

    prompt = f"""You are an Active Learning expert. Compare these AL strategies based on experiment results:

{summary}

Analyze:
1. Which strategy is most efficient (best quality with fewest labels)?
2. How many labeled examples does the best strategy save vs random?
3. Practical recommendation: which strategy and why?

Respond in Russian, structured and concise."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"LLM error: {e}"
```

## Интеграция

### В main.py добавь после сравнения стратегий:

```python
# Бонус: LLM analysis
print("\n" + "=" * 60)
print("BONUS: LLM ANALYSIS")
print("=" * 60)
recommendation = agent.llm_recommend_strategy(histories)
print(recommendation)
```

### В ноутбук добавь ячейку после секции 10:

```python
## Бонус: LLM-анализ стратегий
recommendation = agent.llm_recommend_strategy({
    'entropy': hist_entropy,
    'margin': hist_margin,
    'random': hist_random
})
print(recommendation)
```

### В README добавь:

```markdown
## Бонус: LLM в пайплайне (+1 балл)

- `llm_explain_selection()` — Claude объясняет почему выбранные примеры информативны
- `llm_recommend_strategy()` — Claude сравнивает стратегии и даёт рекомендацию

Требуется: `export ANTHROPIC_API_KEY=sk-...`
```

## Graceful degradation
- anthropic не установлен → строка-сообщение
- API key не задан → строка-сообщение
- API error → строка с ошибкой, без краша

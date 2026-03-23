# Спецификация разметки: sentiment_classification

## Задача
Sentiment classification текстовых отзывов. Задача — определить общий эмоциональный тон текста (positive/negative).

## Классы

### positive
**Определение:** Текст выражает положительное отношение, одобрение, удовлетворение.
**Примеры:**
1. "I had heard this movie was good from a lot of my friends that saw it, and they all said it was amazing, so I had very high expectations- and Nancy Drew exceeded those high expectations! It had funny p..."
2. "I did not have too much interest in watching The Flock.Andrew Lau co-directed the masterpiece trilogy of Infernal Affairs but he had been fired from The Flock and he had been replaced by an emergency ..."
3. "Man, some of you people have got to chill. This movie was artistic genius. Instead of searching for reasoning or messages to justify it in your reality, why can't you understand that it is a work of f..."

### negative
**Определение:** Текст выражает отрицательное отношение, критику, неудовлетворённость.
**Примеры:**
1. "I just saw this at the 2006 Vancouver international film festival. The synopsis in the festival guide sounded pretty good so we decided to check this one out. I'm sorry to say I was very disappointed...."
2. "First of all, the actor they have to play Jesus has blue eyes... half the actors they have playing Jews have blue eyes. Aren't there enough brown-eyed actors out there? Jesus being depicted as having ..."
3. "This movie appears to have been an on the job training exercise for the Coppola family. It doesn't seem to know whether to be an "A" or a "B" western. I mean, the hero is called Hopalong Cassidy for G..."

## Граничные случаи
1. **Смешанные отзывы:** "The acting was great but the plot was terrible" — размечать по общему тону
2. **Ирония/сарказм:** "Oh great, another masterpiece..." — negative (несмотря на "great")
3. **Нейтральные:** "The movie was 2 hours long" — если нет явного тона, смотреть контекст
4. **Короткие тексты:** Одно слово/фраза — опираться на коннотацию

## Инструкция для разметчика
1. Прочитайте весь текст целиком
2. Определите общий тон (positive/negative)
3. При неуверенности — отметьте как "uncertain" в комментариях
4. Время на один пример: ~10-15 секунд

## LabelStudio Config

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single" showInline="true">
    <Choice value="positive"/>
    <Choice value="negative"/>
  </Choices>
</View>
```

Вставьте этот XML в LabelStudio → Project Settings → Labeling Interface → Code.
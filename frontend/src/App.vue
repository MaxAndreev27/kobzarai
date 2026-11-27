<script setup lang="ts">
import { computed, ref } from 'vue'

type RetrievalMethod = 'lexical' | 'hybrid'

interface Source {
  id: number
  title: string
  url: string
  start_char: number
  end_char: number
  text: string
}

interface ChatResponse {
  experimental: true
  status: 'generated' | 'ungrounded' | 'sources' | 'refused'
  retrieval_status: string
  retrieval_method: RetrievalMethod
  answer: string
  sources: Source[]
  model: string | null
  corpus_sha256: string
}

const question = ref('')
const loading = ref(false)
const error = ref('')
const result = ref<ChatResponse | null>(null)
const submittedQuestion = ref('')

const examples = [
  'Яку річку Україна називає братом у «Розритій могилі»?',
  'З якого твору рядки «Не вмирає душа наша, не вмирає воля»?',
]

const canSubmit = computed(() => question.value.trim().length > 0 && !loading.value)
const responseHeading = computed(() => {
  if (result.value?.retrieval_status === 'term_not_found') return 'Збігів не знайдено'
  if (result.value?.status === 'sources') return 'Знайдені фрагменти'
  if (result.value?.status === 'refused') return 'Потрібне уточнення'
  if (result.value?.status === 'ungrounded') return 'Відповідь без джерел'
  return 'Відповідь'
})

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isSource(value: unknown): value is Source {
  return isRecord(value)
    && typeof value.id === 'number'
    && typeof value.title === 'string'
    && typeof value.url === 'string'
    && typeof value.start_char === 'number'
    && typeof value.end_char === 'number'
    && typeof value.text === 'string'
}

function isChatResponse(value: unknown): value is ChatResponse {
  return isRecord(value)
    && value.experimental === true
    && (value.status === 'generated' || value.status === 'ungrounded' || value.status === 'sources' || value.status === 'refused')
    && typeof value.retrieval_status === 'string'
    && (value.retrieval_method === 'lexical' || value.retrieval_method === 'hybrid')
    && typeof value.answer === 'string'
    && Array.isArray(value.sources)
    && value.sources.every(isSource)
    && (value.model === null || typeof value.model === 'string')
    && typeof value.corpus_sha256 === 'string'
}

function sourceUrl(url: string): string | undefined {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : undefined
  } catch {
    return undefined
  }
}

async function submitQuestion() {
  if (!canSubmit.value) return

  const prompt = question.value.trim()
  loading.value = true
  error.value = ''
  result.value = null
  submittedQuestion.value = prompt

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: prompt,
        mode: 'auto',
        retrieval_method: 'hybrid',
        answer_mode: 'generated',
        allow_ungrounded: true,
      }),
    })
    if (!response.ok) {
      const details: Record<number, string> = {
        422: 'Перевірте запит і вибрані налаштування.',
        502: 'Не вдалося отримати коректну відповідь від Ollama.',
        504: 'Час очікування Ollama вичерпано.',
      }
      throw new Error(`Помилка сервера (${response.status}). ${details[response.status] ?? 'Спробуйте ще раз.'}`)
    }
    const data: unknown = await response.json()
    if (!isChatResponse(data)) throw new Error('Сервер повернув відповідь невідомого формату.')
    result.value = data
  } catch (caught) {
    if (caught instanceof SyntaxError) {
      error.value = 'Сервер повернув некоректний JSON.'
    } else if (caught instanceof TypeError) {
      error.value = 'Не вдалося з’єднатися з локальним сервером.'
    } else {
      error.value = caught instanceof Error ? caught.message : 'Сталася невідома помилка.'
    }
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="page">
    <header class="masthead">
      <a class="brand" href="#top" aria-label="Кобзар AI — на початок">
        <span class="brand-mark" aria-hidden="true">✣</span>
        <span>КОБЗАР<span class="brand-accent"> AI</span></span>
      </a>
      <div class="masthead-note"><span class="status-dot"></span> ЕКСПЕРИМЕНТАЛЬНИЙ ПРОЄКТ</div>
      <a class="header-link" href="#about">Про проєкт <span aria-hidden="true">↗</span></a>
    </header>

    <main id="top" class="main">
      <div class="ornament-rule" aria-hidden="true"><span>✦</span></div>
      <section class="hero" aria-labelledby="page-title">
        <div class="hero-copy">
          <div class="eyebrow"><span class="eyebrow-line"></span> СЛОВО, ЩО ЄДНАЄ ЧАСИ</div>
          <h1 id="page-title">Розмова<br />з <em>Кобзарем</em></h1>
          <p class="hero-description">
            Простір для знайомства з поезією Тараса Шевченка.
            Запитуйте про твори, знаходьте рядки та відкривайте їх наново.
          </p>
          <div class="hero-signature"><span aria-hidden="true">✳</span> СЛОВО МАЄ ПАМ’ЯТЬ</div>
        </div>
        <figure class="portrait">
          <div class="portrait-frame">
            <img src="/shevchenko-portrait.jpg" alt="Архівна фотографія Тараса Шевченка" />
          </div>
          <figcaption>ТАРАС ШЕВЧЕНКО <span>·</span> XIX СТОЛІТТЯ</figcaption>
        </figure>
        <div class="hero-aside" aria-hidden="true"><span>ДО ДЖЕРЕЛА</span><i></i><span>З ПОВАГОЮ</span></div>
      </section>

      <section class="workspace" aria-labelledby="workspace-title">
        <div class="section-heading">
          <div>
            <div class="eyebrow"><span class="red-diamond">◆</span> ВАШЕ ЗАПИТАННЯ</div>
            <h2 id="workspace-title">Про що поговоримо?</h2>
          </div>
          <div class="section-number" aria-hidden="true">01 <span>/</span> 02</div>
        </div>

        <div class="notice" id="about">
          <span class="notice-icon" aria-hidden="true">✳</span>
          <p>
            Це ШІ-співрозмовник, натхненний Шевченком, а не історична особа.
            Відповіді можуть містити помилки — звіряйте їх із джерелами.
            Кожне запитання обробляється окремо, без пам’яті попередніх.
          </p>
        </div>

        <form class="question-form" @submit.prevent="submitQuestion">
          <label class="input-label" for="question">Ваше запитання <span>✦</span></label>
          <div class="input-wrap">
            <textarea
              id="question"
              v-model="question"
              :disabled="loading"
              maxlength="4000"
              rows="3"
              placeholder="Наприклад, про що йдеться у «Заповіті»?"
              aria-describedby="input-hint"
            ></textarea>
            <div class="input-bottom">
              <span id="input-hint">Для пошуку в корпусі вкажіть назву твору або цитату в лапках (щонайменше чотири слова).</span>
              <span>{{ question.length }} / 4000</span>
            </div>
          </div>

          <div class="examples">
            <span>СПРОБУЙТЕ ЗАПИТАТИ</span>
            <button v-for="example in examples" :key="example" type="button" :disabled="loading" @click="question = example">
              {{ example }} <span aria-hidden="true">↗</span>
            </button>
          </div>

          <div class="form-footer">
            <span class="footer-hint"><span aria-hidden="true">✧</span> ПОШУК У ТЕКСТАХ І ВІДПОВІДЬ ШІ</span>
            <button class="submit-button" type="submit" :disabled="!canSubmit">
              {{ loading ? 'Шукаємо відповідь…' : 'Запитати' }}
              <span aria-hidden="true">{{ loading ? '···' : '↗' }}</span>
            </button>
          </div>
        </form>

        <div v-if="loading" class="feedback" role="status">Шукаємо фрагменти та готуємо відповідь. Це може зайняти кілька хвилин.</div>
        <div v-if="error" class="feedback feedback-error" role="alert">
          <strong>Не вдалося виконати запит.</strong> {{ error }}
          Перевірте, чи запущені FastAPI та Ollama.
        </div>

        <section v-if="result" class="result" aria-live="polite" aria-labelledby="result-title">
          <div class="result-heading">
            <div><div class="eyebrow"><span class="red-diamond">◆</span> РЕЗУЛЬТАТ</div><h2 id="result-title">{{ responseHeading }}</h2></div>
            <span class="section-number" aria-hidden="true">02 <span>/</span> 02</span>
          </div>
          <p class="result-question">{{ submittedQuestion }}</p>
          <p class="result-answer">{{ result.answer }}</p>
          <p v-if="result.status === 'generated'" class="result-warning">Відповідь згенерована ШІ та може містити помилки. Знайдені фрагменти не обов’язково містять відповідь — перевірте їх самостійно.</p>
          <p v-if="result.status === 'ungrounded'" class="result-warning">Джерело не знайдено або не вдалося визначити однозначно. Це загальна відповідь ШІ без підтвердження корпусом; вона може містити помилки.</p>
          <p v-if="result.sources.length" class="sources-heading">ЗНАЙДЕНІ ФРАГМЕНТИ <span>{{ String(result.sources.length).padStart(2, '0') }}</span></p>
          <details v-for="(source, index) in result.sources" :key="`${source.id}-${source.start_char}-${source.end_char}`" class="source" open>
            <summary><span><span class="source-number">{{ String(index + 1).padStart(2, '0') }}</span> {{ source.title }}</span><span class="settings-plus" aria-hidden="true">+</span></summary>
            <div class="source-content">
              <p>{{ source.text }}</p>
              <a v-if="sourceUrl(source.url)" :href="sourceUrl(source.url)" target="_blank" rel="noopener noreferrer">Відкрити джерело ↗</a>
              <span v-else class="source-link-error">Посилання на джерело недоступне.</span>
            </div>
          </details>
        </section>
      </section>
    </main>

    <footer class="site-footer">
      <div><strong>КОБЗАР<span> AI</span></strong><p>Слово минулого. Розмова сьогодні.</p></div>
      <div class="footer-credit">ПОРТРЕТ: АНДРІЙ ДЕНЬЄР · <a href="https://commons.wikimedia.org/wiki/File:%D0%A2._%D0%93._%D0%A8%D0%B5%D0%B2%D1%87%D0%B5%D0%BD%D0%BA%D0%BE._%D0%9A%D0%B2%D1%96%D1%82%D0%B5%D0%BD%D1%8C_1859.jpg" target="_blank" rel="noopener noreferrer">WIKIMEDIA COMMONS ↗</a></div>
      <div class="footer-ornament" aria-hidden="true">✣ &nbsp; ✣ &nbsp; ✣</div>
    </footer>
  </div>
</template>

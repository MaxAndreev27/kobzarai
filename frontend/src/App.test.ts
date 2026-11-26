import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App.vue'

const response = {
  experimental: true,
  status: 'generated',
  retrieval_status: 'matched',
  retrieval_method: 'lexical',
  answer: 'Перевірте фрагменти.',
  sources: [{
    id: 1,
    title: 'Твір',
    url: 'https://example.org/work',
    start_char: 0,
    end_char: 14,
    text: 'Текст джерела.',
  }],
  model: null,
  corpus_sha256: 'hash',
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('chat form', () => {
  it('uses automatic search and displays the returned passage expanded', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetchMock)
    const wrapper = mount(App)

    await wrapper.get('textarea').setValue('Питання про твір')
    await wrapper.get('form').trigger('submit')
    await flushPromises()

    expect(fetchMock).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        question: 'Питання про твір',
        mode: 'auto',
        retrieval_method: 'hybrid',
        answer_mode: 'generated',
        allow_ungrounded: true,
      }),
    }))
    expect(wrapper.get('.result-answer').text()).toBe('Перевірте фрагменти.')
    expect(wrapper.get('.source-content').text()).toContain('Текст джерела.')
    expect(wrapper.get('.source-content a').attributes('href')).toBe('https://example.org/work')
    expect(wrapper.get('.source').attributes('open')).toBeDefined()
    expect(wrapper.get('.result-warning').text()).toContain('не обов’язково містять відповідь')
    expect(wrapper.find('select').exists()).toBe(false)
  })

  it('clearly identifies an answer not supported by sources', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ...response,
        status: 'ungrounded',
        retrieval_status: 'not_found',
        answer: 'Загальна відповідь.',
        sources: [],
        model: 'qwen3:8b',
      }),
    }))
    const wrapper = mount(App)
    await wrapper.get('textarea').setValue('Як справи?')
    await wrapper.get('form').trigger('submit')
    await flushPromises()

    expect(wrapper.get('#result-title').text()).toBe('Відповідь без джерел')
    expect(wrapper.get('.result-answer').text()).toBe('Загальна відповідь.')
    expect(wrapper.get('.result-warning').text()).toContain('без підтвердження корпусом')
    expect(wrapper.find('.source').exists()).toBe(false)
  })

  it('shows HTTP and invalid response errors rather than a result', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 502 })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ answer: 'invalid' }) })
    vi.stubGlobal('fetch', fetchMock)
    const wrapper = mount(App)
    await wrapper.get('textarea').setValue('Питання')

    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('502')
    expect(wrapper.find('.result').exists()).toBe(false)

    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('невідомого формату')
    expect(wrapper.find('.result').exists()).toBe(false)
  })
})

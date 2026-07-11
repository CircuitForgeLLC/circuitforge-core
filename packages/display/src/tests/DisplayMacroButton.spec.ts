import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DisplayMacroButton from '../components/DisplayMacroButton.vue'

describe('DisplayMacroButton', () => {
  it('renders icon and label', () => {
    const wrapper = mount(DisplayMacroButton, {
      props: { icon: '🔄', label: 'Restart nginx', action: { type: 'shell', command: 'systemctl restart nginx' } },
    })
    expect(wrapper.text()).toContain('🔄')
    expect(wrapper.text()).toContain('Restart nginx')
  })

  it('emits trigger with the action payload on click', async () => {
    const action = { type: 'shell' as const, command: 'systemctl restart nginx' }
    const wrapper = mount(DisplayMacroButton, {
      props: { icon: '🔄', label: 'Restart nginx', action },
    })
    await wrapper.trigger('click')
    expect(wrapper.emitted('trigger')).toHaveLength(1)
    expect(wrapper.emitted('trigger')![0]).toEqual([action])
  })

  it('does not emit trigger when disabled', async () => {
    const wrapper = mount(DisplayMacroButton, {
      props: {
        icon: '🔄',
        label: 'Restart nginx',
        action: { type: 'shell', command: 'x' },
        disabled: true,
      },
    })
    await wrapper.trigger('click')
    expect(wrapper.emitted('trigger')).toBeUndefined()
  })

  it('sets the disabled attribute on the button element', () => {
    const wrapper = mount(DisplayMacroButton, {
      props: {
        icon: '🔄',
        label: 'x',
        action: { type: 'shell', command: 'x' },
        disabled: true,
      },
    })
    expect(wrapper.attributes('disabled')).toBeDefined()
  })

  it('supports url actions', async () => {
    const action = { type: 'url' as const, url: 'https://example.com' }
    const wrapper = mount(DisplayMacroButton, {
      props: { icon: '🌐', label: 'Docs', action },
    })
    await wrapper.trigger('click')
    expect(wrapper.emitted('trigger')![0]).toEqual([action])
  })

  it('supports display_switch actions', async () => {
    const action = { type: 'display_switch' as const, target: 'robin' }
    const wrapper = mount(DisplayMacroButton, {
      props: { icon: '↔️', label: 'Switch', action },
    })
    await wrapper.trigger('click')
    expect(wrapper.emitted('trigger')![0]).toEqual([action])
  })

  it('sets an accessible label matching the button label', () => {
    const wrapper = mount(DisplayMacroButton, {
      props: { icon: '🔄', label: 'Restart nginx', action: { type: 'shell', command: 'x' } },
    })
    expect(wrapper.attributes('aria-label')).toBe('Restart nginx')
  })
})

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DisplayAlert from '../components/DisplayAlert.vue'

describe('DisplayAlert', () => {
  it('renders the message', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'pacman lock detected', timestamp: '2026-01-01T14:22:00' },
    })
    expect(wrapper.text()).toContain('pacman lock detected')
  })

  it('formats an ISO timestamp as a time string', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: '2026-01-01T14:22:00' },
    })
    expect(wrapper.find('.cf-display-alert__time').text()).toMatch(/\d{1,2}:\d{2}/)
  })

  it('accepts a Date object for timestamp', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: new Date('2026-01-01T14:22:00') },
    })
    expect(wrapper.find('.cf-display-alert__time').text()).toMatch(/\d{1,2}:\d{2}/)
  })

  it('falls back to the raw string when timestamp is unparseable', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: 'not-a-date' },
    })
    expect(wrapper.find('.cf-display-alert__time').text()).toBe('not-a-date')
  })

  it('defaults severity to ok', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: '2026-01-01T00:00:00' },
    })
    expect(wrapper.classes()).toContain('cf-display-alert--ok')
  })

  it('applies the warn severity class', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: '2026-01-01T00:00:00', severity: 'warn' },
    })
    expect(wrapper.classes()).toContain('cf-display-alert--warn')
  })

  it('applies the crit severity class', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: '2026-01-01T00:00:00', severity: 'crit' },
    })
    expect(wrapper.classes()).toContain('cf-display-alert--crit')
  })

  it('has an accessible status role', () => {
    const wrapper = mount(DisplayAlert, {
      props: { message: 'x', timestamp: '2026-01-01T00:00:00' },
    })
    expect(wrapper.attributes('role')).toBe('status')
  })
})

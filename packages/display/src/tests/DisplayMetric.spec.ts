import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DisplayMetric from '../components/DisplayMetric.vue'

describe('DisplayMetric', () => {
  it('renders value and label', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 87, label: 'CPU temp' },
    })
    expect(wrapper.text()).toContain('87')
    expect(wrapper.text()).toContain('CPU temp')
  })

  it('appends unit to value when provided', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 87, label: 'CPU temp', unit: '°C' },
    })
    expect(wrapper.find('.cf-display-metric__value').text()).toBe('87°C')
  })

  it('does not append anything when unit is omitted', () => {
    const wrapper = mount(DisplayMetric, { props: { value: 42, label: 'Fans' } })
    expect(wrapper.find('.cf-display-metric__value').text()).toBe('42')
  })

  it('defaults severity to ok', () => {
    const wrapper = mount(DisplayMetric, { props: { value: 1, label: 'x' } })
    expect(wrapper.classes()).toContain('cf-display-metric--ok')
  })

  it('applies the warn severity class', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 1, label: 'x', severity: 'warn' },
    })
    expect(wrapper.classes()).toContain('cf-display-metric--warn')
  })

  it('applies the crit severity class', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 1, label: 'x', severity: 'crit' },
    })
    expect(wrapper.classes()).toContain('cf-display-metric--crit')
  })

  it('does not render a sparkline when none is given', () => {
    const wrapper = mount(DisplayMetric, { props: { value: 1, label: 'x' } })
    expect(wrapper.find('.cf-display-metric__sparkline').exists()).toBe(false)
  })

  it('does not render a sparkline with fewer than 2 points', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 1, label: 'x', sparkline: [5] },
    })
    expect(wrapper.find('.cf-display-metric__sparkline').exists()).toBe(false)
  })

  it('renders a sparkline polyline with one point per reading', () => {
    const wrapper = mount(DisplayMetric, {
      props: { value: 1, label: 'x', sparkline: [1, 2, 3, 2, 1] },
    })
    const polyline = wrapper.find('polyline')
    expect(polyline.exists()).toBe(true)
    const points = polyline.attributes('points')!.trim().split(' ')
    expect(points).toHaveLength(5)
  })

  it('sets an accessible group label matching the metric label', () => {
    const wrapper = mount(DisplayMetric, { props: { value: 1, label: 'RAM usage' } })
    expect(wrapper.attributes('aria-label')).toBe('RAM usage')
  })
})

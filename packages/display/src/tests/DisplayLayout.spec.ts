import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DisplayLayout from '../components/DisplayLayout.vue'

describe('DisplayLayout', () => {
  it('renders the product name in the identity zone by default', () => {
    const wrapper = mount(DisplayLayout, { props: { product: 'turnstone' } })
    expect(wrapper.find('.cf-display-layout__product').text()).toBe('turnstone')
  })

  it('defaults profile to sysadmin', () => {
    const wrapper = mount(DisplayLayout, { props: { product: 'turnstone' } })
    expect(wrapper.find('.cf-display-layout__profile').text()).toBe('sysadmin')
    expect(wrapper.classes()).toContain('cf-display-layout--profile-sysadmin')
  })

  it('applies the given profile', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'robin', profile: 'casual' },
    })
    expect(wrapper.classes()).toContain('cf-display-layout--profile-casual')
  })

  it('defaults orientation to landscape', () => {
    const wrapper = mount(DisplayLayout, { props: { product: 'turnstone' } })
    expect(wrapper.classes()).toContain('cf-display-layout--landscape')
  })

  it('applies portrait orientation when set', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone', orientation: 'portrait' },
    })
    expect(wrapper.classes()).toContain('cf-display-layout--portrait')
  })

  it('renders content passed to the metrics slot', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone' },
      slots: { metrics: '<div class="probe-metrics">m</div>' },
    })
    expect(wrapper.find('.cf-display-layout__metrics .probe-metrics').exists()).toBe(true)
  })

  it('renders content passed to the alerts slot', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone' },
      slots: { alerts: '<div class="probe-alerts">a</div>' },
    })
    expect(wrapper.find('.cf-display-layout__alerts .probe-alerts').exists()).toBe(true)
  })

  it('renders content passed to the macros slot', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone' },
      slots: { macros: '<div class="probe-macros">x</div>' },
    })
    expect(wrapper.find('.cf-display-layout__macros .probe-macros').exists()).toBe(true)
  })

  it('overrides the identity zone with the identity slot', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone' },
      slots: { identity: '<span class="probe-identity">Custom</span>' },
    })
    expect(wrapper.find('.probe-identity').exists()).toBe(true)
    expect(wrapper.find('.cf-display-layout__product').exists()).toBe(false)
  })

  it('emits open-launcher when the identity button is tapped', async () => {
    const wrapper = mount(DisplayLayout, { props: { product: 'turnstone' } })
    await wrapper.find('.cf-display-layout__identity').trigger('click')
    expect(wrapper.emitted('open-launcher')).toHaveLength(1)
  })

  it('defaults to dark theme and sets the data attribute', () => {
    const wrapper = mount(DisplayLayout, { props: { product: 'turnstone' } })
    expect(wrapper.attributes('data-cf-display-theme')).toBe('dark')
  })

  it('applies light theme override values', () => {
    const wrapper = mount(DisplayLayout, {
      props: { product: 'turnstone', theme: 'light' },
    })
    expect(wrapper.attributes('data-cf-display-theme')).toBe('light')
    expect(wrapper.attributes('style')).toContain('#ffffff')
  })
})

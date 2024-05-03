<template>
    <framed-card>
      <template #header>
        <p>
          {{ nt(colorVar.name) || $t('core.hmi.general.materials.addColor') }}
        </p>
      </template>
  
      <template #content>
        <v-card>
          <v-card-title> Color Details </v-card-title>
          <v-card-text>
            <v-form style="" class="custom-form">
              <v-row>
                <v-col cols="3">
                  <span class="color-name-title">Name</span>
                </v-col>
                <v-col v-if="hasValidRal(colorVar.id)">
                  <span class="ral-code">
                    {{ colorVar.ralCode }}
                  </span>
                </v-col>
                <v-col
                  :cols="hasValidRal(colorVar.id) ? '6' : '6'"
                  :class="{ 'align-right': !hasValidRal(colorVar.ralCode) }"
                >
                  <v-text-field
                    v-model="colorVar.name.en"
                    v-touch-keyboard
                    outlined
                  >
                  </v-text-field>
                </v-col>
              </v-row>
  
              <v-row>
                <v-col>
                  <span class="hex-name-title">Hex Value</span>
                </v-col>
                <v-col>
                  <v-text-field v-model="colorVar.hex" v-touch-keyboard outlined>
                  </v-text-field>
                </v-col>
              </v-row>
  
              <v-row>
                <v-col>
                  <span style="position: relative; top: 5px">RGB Value</span>
                </v-col>
                <v-col>
                  <v-text-field v-model="rgbValues.r" v-touch-keyboard @input="updateRgbValues" min="0" max="255">
                  </v-text-field>
                </v-col>
                <v-col>
                  <v-text-field v-model="rgbValues.g" v-touch-keyboard @input="updateRgbValues">
                  </v-text-field>
                </v-col>
                <v-col>
                  <v-text-field v-model="rgbValues.b" v-touch-keyboard @input="updateRgbValues">
                  </v-text-field>
                </v-col>
              </v-row>
              <br />
              <v-row class="d-flex justify-center pt-10" style="">
                <v-color-picker v-model="colorVar.hex" mode="rgba" hide-inputs>
                </v-color-picker>
              </v-row>
            </v-form>
          </v-card-text>
        </v-card>
      </template>
      <template #actions>
        <v-btn color="white" @click="closeDetail">
          <v-icon color="red">mdi-close</v-icon>
          {{ $t('core.hmi.general.buttons.cancel') }}
        </v-btn>
        <v-btn color="white" @click="saveColor()">
          <v-icon color="green">mdi-check</v-icon>
          {{ $t('core.hmi.general.buttons.save') }}
        </v-btn>
      </template>
    </framed-card>
  </template>
  
  <script setup lang="ts">
    import { defineEmits, defineProps, ref, computed } from 'vue'
    import { Color } from '@/types/materials/Color'
    import { useColorStore } from '@/store/materials/Color'
    import FramedCard from '@/components/general/FramedCard.vue'
    import { useNameTranslation } from '@/core/composables/useNameTranslation'
    import ARRAY from '@/core/helper/array'
    import copy from '@/core/helper/copy'
    import { useNotificationStore } from '@/store/notifications/Notifications'
  
    const { nt } = useNameTranslation()
    const colorStore = useColorStore()
    const notificationStore = useNotificationStore()
  
    Promise.all([colorStore.getAll()]).finally(() => {
      // console.log(colorStore.colorArray[0])
    })
  
    const props = defineProps<{
      modelValue: Color
    }>()
  
    const emit = defineEmits<{
      (event: 'closeDetail'): void
    }>()
  
    const colorVar = ref(props.modelValue)
  
    const closeDetail = () => {
      emit('closeDetail')
    }
  
    const translatedName = computed({
      get: () => nt(colorVar.value.name),
      set: (newValue) => {
        colorVar.value.name.en = newValue
      },
    })
  
    const saveColor = () => {
      if (colorVar.value.id.length > 0) {
        let currentEntity = ARRAY.getById<Color>(
          colorStore.colorArray,
          colorVar.value.id
        )
        if (currentEntity !== null) {
          colorStore
            .update(Object.assign(copy(currentEntity), colorVar.value))
            .catch((error) => {
              notificationStore.parseErrors(error.response?.data)
            })
        }
      } else {
        colorStore.add(colorVar.value).catch((error) => {
          notificationStore.parseErrors(error.response?.data)
        })
      }
      emit('closeDetail')
    }
  
    const rgbValues = computed({
      get() {
        // Existing code to extract RGB from Hex
        return hexToRgb(colorVar.value.hex)
      },
      set(values) {
        // Convert RGB back to Hex and update colorVar.hex
        colorVar.value.hex = rgbToHex(values.r, values.g, values.b)
      },
    })
  
    function rgbToHex(r: string | number, g: string | number, b: string | number) {
      // Ensure the RGB values are numbers and within the valid range (0-255)
      r = Math.max(0, Math.min(255, parseInt(r, 10)));
      g = Math.max(0, Math.min(255, parseInt(g, 10)));
      b = Math.max(0, Math.min(255, parseInt(b, 10)));
  
      // Convert each RGB value to a hexadecimal string
      const hexR = r.toString(16).padStart(2, '0');
      const hexG = g.toString(16).padStart(2, '0');
      const hexB = b.toString(16).padStart(2, '0');
  
      // Combine the hexadecimal strings to form the Hex value
      return `#${hexR}${hexG}${hexB}`
    }
  
  
    function hexToRgb(hex: string | never[]) {
      let r = 0,
        g = 0,
        b = 0
  
      //   3 digits
      if (hex.length === 4) {
        r = parseInt(hex[1] + hex[1], 16)
        g = parseInt(hex[2] + hex[2], 16)
        b = parseInt(hex[3] + hex[3], 16)
      } else if (hex.length === 7) {
        r = parseInt(hex[1] + hex[2], 16)
        g = parseInt(hex[3] + hex[4], 16)
        b = parseInt(hex[5] + hex[6], 16)
      }
      return { r, g, b }
    }
  
    const hasValidRal = (ral: any) => {
      return /^RAL\d{4}$/.test(ral)
    }
  
    const updateRgbValues = () => {
      // Manually trigger the set method of the rgbValues computed property
      rgbValues.value = { r: rgbValues.value.r, g: rgbValues.value.g, b: rgbValues.value.b };
    };
  </script>
  
  <style scoped lang="sass">
  
    .custom-form
      height: 700px
      overflow-y: auto
      padding-bottom: 0
      padding-top: 10px
      overflow-x: hidden
  
    .color-name-title
      align-self: center
      position: relative
      top: 8px
  
    .ral-code
      align-self: center
      justify-content: center
      position: relative
      top: 10px
      margin-left: 35px
  
    .hex-name-title
      align-self: center
      position: relative
      top: 8px
  
    .align-right
      position: relative
      right: 0px
      align-self: center
      align-content: flex-end
      margin-left: 140px
  </style>
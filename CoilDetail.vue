<template>
    <v-row class="h-100">
      <v-col class="h-100">
        <keyboard-scrollview>
          <v-row class="mt-3 ml-4 mb-3"
            ><h1>{{ $t('module.coilfarm.state.hmi.general') }}</h1></v-row
          >
          <v-row v-if="!props.coil.isEmpty" class="mb-4 ml-1"
            ><v-col cols="4">
              <color-display
                :front-color="getMaterialColor(selectedMaterial.color.frontId)"
                :back-color="getMaterialColor(selectedMaterial.color.backId)"
                :rounded="false"
              ></color-display></v-col
            ><v-col cols="7"
              ><v-row class="mb-2"
                ><b>{{ $t('core.hmi.general.rawMaterials.singular') }}&nbsp;</b>
                <p v-if="selectedMaterial.rawMaterialId">
                  {{
                    nt(
                      rawMaterialStore.getRawMaterialById(
                        selectedMaterial.rawMaterialId
                      ).name
                    )
                  }}
                </p></v-row
              ><v-row class="mb-1"
                ><b>{{ $t('core.hmi.general.materials.coating') }}&nbsp;</b>
                <p v-if="selectedMaterial.coatingId">
                  {{
                    nt(
                      coatingStore.getCoatingById(selectedMaterial.coatingId).name
                    )
                  }}
                </p></v-row
              ><v-row class="mb-1"
                ><b>{{ $t('core.hmi.general.materials.thickness') }}&nbsp;</b>
                <p>{{ selectedMaterial.thickness }}</p></v-row
              ><v-row class="mb-1"
                ><b>{{ $t('core.hmi.general.materials.foil') }}&nbsp;</b>
                <p>{{ selectedMaterial.hasFoil ? 'yes' : 'no' }}</p></v-row
              ><!--<v-row class="mb-1"
                ><b>{{ $t('core.hmi.general.materials.perforated') }}&nbsp;</b>
                <p>
                  {{ selectedMaterial.isPerforated ? 'yes' : 'no' }}
                </p></v-row
              >--><v-row class="mb-1"
                ><b>{{ $t('core.hmi.general.materials.color.front') }}&nbsp;</b>
                <p>
                  {{
                    nt(
                      colorStore.getColorById(selectedMaterial.color.frontId).name
                    )
                  }}
                </p></v-row
              ><v-row
                ><b>{{ $t('core.hmi.general.buttons.back') }}&nbsp;</b>
                <p>
                  {{
                    nt(
                      colorStore.getColorById(selectedMaterial.color.backId).name
                    )
                  }}
                </p></v-row
              ></v-col
            >
          </v-row>
  
          <form-input
            v-model="selectedCoilPlace"
            :coil-places="props.coilPlaces"
            :readonly="props.readOnly"
            input-cols="6"
            label-style="ml-5 mt-1"
            label-cols="4"
            @update:modelValue="emit('update:coilPlace', $event)"
            ><template #label>{{
              $t('core.hmi.general.materials.coilPlaces.plural')
            }}</template></form-input
          >
  
          <template v-if="!props.coil.isEmpty">
            <form-input
              v-model.number="selectedCoil.width"
              v-touch-numeric-global="{ onBlur: 'close' }"
              unit="mm"
              :restrictions="
                loadConfigRestriction('core.master.setup.limitation.width')
              "
              :readonly="props.readOnly"
              input-cols="3"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label>{{ $t('core.locale.units.width') }} </template>
            </form-input>
            <form-input
              v-model.number="selectedCoil.thickness"
              v-touch-numeric-global="{ onBlur: 'close' }"
              unit="mm"
              :restrictions="
                loadConfigRestriction('core.master.setup.limitation.thickness')
              "
              :readonly="props.readOnly"
              input-cols="3"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label
                >{{ $t('core.hmi.general.materials.thickness') }}
              </template>
            </form-input>
            <form-input
              v-model.number="selectedCoil.innerDiameter"
              v-touch-numeric-global="{ onBlur: 'close' }"
              unit="mm"
              :restrictions="
                loadConfigRestriction(
                  'core.master.setup.limitation.innerDiameter'
                )
              "
              :readonly="props.readOnly"
              input-cols="3"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label
                >{{ $t('core.hmi.general.materials.coils.innerDiameter') }}
              </template>
            </form-input>
            <form-input
              v-model.number="selectedCoil.outerDiameter"
              v-touch-numeric-global="{ onBlur: 'close' }"
              unit="mm"
              :restrictions="
                loadConfigRestriction(
                  'core.master.setup.limitation.outerDiameter'
                )
              "
              :readonly="props.readOnly"
              input-cols="3"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label
                >{{ $t('core.hmi.general.materials.coils.outerDiameter') }}
              </template>
            </form-input>
            <form-input
              v-model="selectedCoil.batchNumber"
              :readonly="props.readOnly"
              :global-keyboard="true"
              input-cols="6"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label>{{ $t('core.hmi.general.number') }} </template>
            </form-input>
            <form-input
              v-model="selectedCoil.note"
              :readonly="props.readOnly"
              :global-keyboard="true"
              input-cols="6"
              label-style="ml-5 mt-1"
              label-cols="4"
            >
              <template #label
                >{{ $t('core.hmi.general.notes.singular') }}
              </template>
            </form-input>
          </template>
        </keyboard-scrollview>
      </v-col>
      <template v-if="!props.coil.isEmpty">
        <v-divider :vertical="true" />
        <v-col>
          <keyboard-scrollview>
            <v-row class="mt-3 ml-1 mb-3"
              ><h1>
                {{ $t('core.hmi.general.materials.processing.processingData') }}
              </h1></v-row
            >
            <v-row>
              <processing-unit
                :material="selectedMaterial"
                :processing="props.processing"
                class="ml-4 mb-16"
                @open-processing-menu="openProcessingMenu(processingVar)"
              ></processing-unit>
            </v-row>
            <v-row class="mt-3 ml-1 mb-7"
              ><h1>{{ $t('core.hmi.general.usage') }}</h1></v-row
            >
            <v-row>
              <v-col cols="12" class="d-flex align-center">
            <form-input
              v-model.number="selectedCoil.weight.initial"
              :readonly="props.readOnly"
              :global-keyboard="true"
              :restrictions="
                loadConfigRestriction('core.master.setup.limitation.weight')
              "
              input-cols="4"
              unit="kg"
              label-style="ml-2 mt-0"
              label-cols="4"
              class="custom-instance"
            >
              <template #label>
                <span style="display: inline;white-space:nowrap;align-self:center;position:relative;top:4px;">{{
                  $t('core.hmi.general.materials.weight.initial')
                }}</span>
              </template>
            </form-input>
                <span class="ml-2 length-value">(length: {{ (selectedCoil.length.initial / 1000).toFixed(1) }}m)</span>
              </v-col>
  
            <v-col cols="12" class="d-flex align-center">
            <form-input
              v-model.number="selectedCoil.weight.remaining"
              :readonly="props.readOnly"
              :global-keyboard="true"
              input-cols="4"
              unit="kg"
              label-style="ml-2 mt-0"
              label-cols="4"
              class="custom-instance"
            >
              <template #label>
                <span style="display: inline;white-space:nowrap;align-self:center;position:relative;top:4px;">{{
                  $t('core.hmi.general.materials.weight.remaining')
                }}</span>
              </template>
            </form-input>
              <span class="ml-2 length-value">(length: {{ (selectedCoil.length.remaining / 1000).toFixed(1) }}m)</span>
            </v-col>
            </v-row>
  
            <v-row class="ml-2 mr-10">
              <v-progress-linear
                height="12"
                bg-color="black"
                :model-value="
                  coilStore.remainingCoilWeightPercentage(selectedCoil)
                "
                :color="
                  coilStore.remainingCoilWeightPercentage(selectedCoil) <= 20
                    ? '#B00020'
                    : '#a0c517'
                "
                class="mt-2"
              />
  
              <span class="remaining-coil-weight-text"
                >{{
                  (coilStore.remainingCoilWeightPercentage(props.coil) * 10000) /
                  1000
                }}{{ 'm ' + $t('core.hmi.general.available') }}</span
              >
              <br>
  <!--            <span class="" style="font-size:13px;font-weight:400;text-align:right;">-->
  <!--              {{ (coilStore.initialCoilWeightPercentage(props.coil)) }}-->
  <!--            </span>-->
            </v-row>
          </keyboard-scrollview>
        </v-col>
      </template>
      <template v-if="coilActions == true">
        <v-divider :vertical="true" />
  
        <v-col cols="4">
          <v-row class="mt-3 ml-1 mb-1"
            ><h1>{{ $t('module.coilfarm.hmi.action') }}</h1></v-row
          >
          <v-row v-if="showAction(actionConfig?.removeCoilFromCoilPlace)"
            ><v-col cols="3"
              ><img
                width="70"
                height="70"
                class="ml-1"
                :src="'/icons/actions/actions_default.svg'"
                @click="removeCoilFromPlace" /></v-col
            ><v-col class="mt-4">{{
              $t('module.coilfarm.actions.moveOut')
            }}</v-col></v-row
          >
          <v-row
            v-if="showAction(actionConfig?.coilOnDecoiler) && !props.coil.isEmpty"
            ><v-col cols="3"
              ><img
                width="70"
                height="70"
                class="ml-1"
                :src="'/icons/actions/actions_default.svg'"
                @click="coilOnDecoiler" /></v-col
            ><v-col class="mt-4">{{
              $t('module.coilfarm.actions.threadIn')
            }}</v-col></v-row
          >
          <v-row v-if="showAction(actionConfig?.threadOut)"
            ><v-col cols="3"
              ><img
                width="70"
                height="70"
                class="ml-1"
                :src="'/icons/actions/actions_default.svg'"
                @click="threadOut" /></v-col
            ><v-col class="mt-4">{{
              $t('module.coilfarm.actions.threadOut')
            }}</v-col></v-row
          >
          <v-row v-if="showAction(actionConfig?.moveToPlace)"
            ><v-col cols="3"
              ><img
                width="70"
                height="70"
                class="ml-1"
                :src="'/icons/actions/actions_default.svg'"
                @click="moveToPlaceDialog = true" /></v-col
            ><v-col class="mt-4">{{
              $t('module.coilfarm.actions.moveToPlace')
            }}</v-col></v-row
          >
        </v-col>
      </template>
    </v-row>
    <root-dialog v-model="processingMenu" content-class="processing-dialog">
      <processing-menu
        v-model="processingVar"
        :coil-id="props.coil.id"
        @closeMenu="closeProcessingMenu()"
      ></processing-menu>
    </root-dialog>
  
    <root-dialog v-model="moveToPlaceDialog" content-class="move-in-dialog">
      <framed-card>
        <template #header>
          <h1>{{ $t('core.hmi.general.buttons.selectPlaceAndConfirm') }}</h1>
        </template>
        <template #content>
          <v-container>
            <v-row class="ml-2 mr-10">
              <v-col cols="4"
                ><b>{{
                  $t('core.hmi.general.materials.coilPlaces.singular')
                }}</b></v-col
              >
              <v-col cols="8"
                ><v-select
                  v-model="moveToCoilPlace"
                  :items="
                    props.coilPlaces
                      .filter((entry) => {
                        return (
                          entry.type !== 'hook' &&
                          !entry.coilId &&
                          entry.type !== 'setup'
                        )
                      })
                      .map((entry) => {
                        return { title: entry.label, value: entry.id }
                      })
                  "
                ></v-select
              ></v-col>
            </v-row>
          </v-container>
        </template>
        <template #actions>
          <text-button
            :text="$t('core.hmi.general.buttons.back')"
            text-color="white"
            icon="mdi-chevron-left"
            icon-color="white"
            icon-size="1.5rem"
            @click="moveToPlaceDialog = false"
          ></text-button>
          <text-button
            :text="$t('core.hmi.general.buttons.confirmAndMove')"
            text-color="white"
            icon="mdi-check"
            icon-color="green"
            icon-size="1.5rem"
            :disabled="moveToCoilPlace === ''"
            @click="moveTo(moveToCoilPlace)"
          />
        </template>
      </framed-card>
    </root-dialog>
  </template>
  <script setup lang="ts">
    import ProcessingUnit from '@/components/material/ProcessingUnit.vue'
    import { useCoilStore } from '@/store/materials/Coil'
    import { Coil } from '@/types/materials/Coils'
    import { Material } from '@/types/materials/Materials'
    import {
      defineEmits,
      defineProps,
      onMounted,
      onUnmounted,
      ref,
      toRef,
      watchEffect,
      watch,
      computed,
    } from 'vue'
    import { Processing } from '@/types/materials/Processing'
    import { useCoatingStore } from '@/store/materials/Coating'
    import { useRawMaterialStore } from '@/store/materials/RawMaterial'
    import { useColorStore } from '@/store/materials/Color'
    import { CoilPlace } from '@/types/coilfarm/CoilPlace'
    import ProcessingMenu from '@/components/material/ProcessingMenu.vue'
    import copy from '@/core/helper/copy'
    import { useUiStore } from '@/store/general/Ui'
    import KeyboardScrollview from '@/components/KeyboardScrollview.vue'
    import ColorDisplay from '@/components/material/ColorDisplay.vue'
    import RootDialog from '@/components/general/RootDialog.vue'
    import TextButton from '@/components/button/TextButton.vue'
    import FramedCard from '@/components/general/FramedCard.vue'
    import FormInput from '@/components/general/FormInput.vue'
    import { useConfigStore } from '@/store/config/Configs'
    import { useCoilFarmState } from '@/store/coilfarm/CoilFarmState'
    import { useNameTranslation } from '@/core/composables/useNameTranslation'
  
    const coilStore = useCoilStore()
    const coatingStore = useCoatingStore()
    const rawMaterialStore = useRawMaterialStore()
    const colorStore = useColorStore()
    const coilStateStore = useCoilFarmState()
    const uiStore = useUiStore()
    const configStore = useConfigStore()
  
    const props = defineProps<{
      coil: Coil
      material: Material
      processing: Processing
      coilPlaces: CoilPlace[]
      coilPlace: string
      coilActions?: boolean
      readOnly?: boolean
      actionConfig?: {
        removeCoilFromCoilPlace?: boolean
        coilOnDecoiler?: boolean
        threadOut?: boolean
        moveToPlace?: boolean
      }
    }>()
    const { nt } = useNameTranslation()
    const selectedCoilPlace = ref(props.coilPlace)
    const moveToPlaceDialog = ref(false)
    const actionConfig = ref(props.actionConfig)
    watch(
      () => props.material,
      (newVal) => {
        selectedMaterial.value = newVal
      }
    )
  
    const showAction = (value: boolean | undefined) => {
      if (!actionConfig.value) {
        return true
      } else if (value === undefined) {
        return true
      } else {
        return value
      }
    }
  
    const emit = defineEmits<{
      (event: 'removeCoilFromPlace'): void
      (event: 'closeDetail'): void
      (event: 'coilOnDecoiler'): void
      (event: 'threadOut'): void
      (event: 'moveToPlace', value: string): void
      (event: 'update:coilPlace', value: string): void
    }>()
  
    const removeCoilFromPlace = () => {
      emit('removeCoilFromPlace')
    }
  
    const coilOnDecoiler = () => {
      emit('coilOnDecoiler')
    }
  
    const threadOut = () => {
      emit('threadOut')
    }
  
    const moveTo = (placeId: string) => {
      emit('moveToPlace', placeId)
    }
  
    const selectedMaterial = ref(props.material)
    const selectedCoil = ref(props.coil)
    const moveToCoilPlace = ref('')
    const processingVar = ref(props.processing)
    const processingMenu = toRef(uiStore, 'isProcessingActive')
  
    watchEffect(() => {
      selectedCoil.value = props.coil
      // selectedCoil.value.outerDiameter = coilStateStore.diameter  //for the outer Diameter if data comes from websocket
    })
  
    function loadConfigRestriction(id: string) {
      const config = configStore.getConfigById(id)
      if (config) {
        return configStore.resolveRestrictions(config.restrictions)
      }
      console.error('There is no config ', id, ' available')
    }
  
    function getMaterialColor(colorId: string) {
      let color = colorStore.getColorById(colorId)
      if (color) {
        return color.hex
      } else return null
    }
  
    function closeProcessingMenu(event?: MouseEvent) {
      let toggleOff
  
      if (event) {
        toggleOff = (event.target as HTMLElement)?.classList.contains(
          'v-overlay__content'
        )
      } else {
        toggleOff = true
      }
  
      if (toggleOff) {
        processingMenu.value = false
        uiStore.isProcessingActive = false
      }
    }
    function openProcessingMenu(processing: Processing) {
      if (processingVar.value.id) {
        processingVar.value = copy(processing)
        processingMenu.value = true
      } else {
        processingVar.value = new Processing()
        processingVar.value.thickness = selectedCoil.value.thickness
        processingVar.value.rawMaterialId = selectedMaterial.value.rawMaterialId
        processingVar.value.coatingId = selectedMaterial.value.coatingId
        processingVar.value.coilId = selectedCoil.value.id
        processingMenu.value = true
      }
    }
  </script>
  <style scoped lang="sass">
    @import '@/styles/sass/RightMenu.sass'
    @import '@/styles/sass/Material.sass'
  </style>
  
  <style lang="sass">
  
    .processing-dialog
      height: 80%
      margin: 0 0 0 0 !important
      right: 0
  
    .length-value
      font-size: 14px
      margin-right: 0px
      align-self: center
      display: inline
      white-space: nowrap
      position: relative
      top: 3px
      font-weight: 400
      left: -70px
  </style>
  
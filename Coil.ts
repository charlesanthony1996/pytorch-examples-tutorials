import { defineStore } from 'pinia'
import { Coil, CoilDTO } from '@/types/materials/Coils'
import GetAll from '@/core/api/GetAll'
import Add from '@/core/api/Create'
import { ref } from 'vue'
import COMMON_API from '@/core/api/CommonApi'
import DATE from '@/core/helper/date'

export const useCoilStore = defineStore('Coilstore', () => {
  const coilArray = ref([])
  const allCoilFarmCoils = ref([])

  function toDTO(entity: Coil): CoilDTO {
    return {
      ...entity,
      createdAt: DATE.toDTO(entity.createdAt),
      updatedAt: DATE.toDTO(entity.updatedAt),
      lastUsedAt: DATE.toDTO(entity.lastUsedAt),
    }
  }

  function toPartialDTO(entity: Partial<Coil>): Partial<CoilDTO> {
    return {
      ...entity,
      createdAt: DATE.toDTO(entity.createdAt),
      updatedAt: DATE.toDTO(entity.updatedAt),
      lastUsedAt: DATE.toDTO(entity.lastUsedAt),
    }
  }

  function fromDTO(entity: CoilDTO): Coil {
    return {
      ...entity,
      createdAt: DATE.fromDTO(entity.createdAt),
      updatedAt: DATE.fromDTO(entity.updatedAt),
      lastUsedAt: DATE.fromDTO(entity.lastUsedAt),
    }
  }

  const commonApi = COMMON_API<Coil, CoilDTO>('/core/material/coils', {
    targetArray: coilArray.value,
    transforms: {
      toDTO,
      toPartialDTO,
      fromDTO,
    },
  })

  const getCoilsForMaterial = GetAll<Coil, CoilDTO>(
    '/core/material/materials/{{id}}/coils',
    {
      //targetArray: coilsForMaterial.value,
      transforms: {
        toDTO,
        toPartialDTO,
        fromDTO,
      },
    }
  )

  const remainingCoilWeightPercentage = (coil: Coil) => {
    return Math.round((coil.weight.remaining * 100) / coil.weight.initial)
  }

  const initialCoilWeightPercentage = (coil: Coil) => {
    return Math.round(coil.weight.initial)
  }

  const addCoilToMaterial = Add<Coil, CoilDTO>(
    '/core/material/materials/{{materialId}}/coils',
    {
      //targetArray: coilsForMaterial.value,
      transforms: {
        toDTO,
        toPartialDTO,
        fromDTO,
      },
    }
  )

  const getAllCoilFarmCoils = GetAll<Coil, CoilDTO>('/module/coilfarm/coils', {
    transforms: {
      toDTO,
      toPartialDTO,
      fromDTO,
    },
    targetArray: allCoilFarmCoils.value,
  })

  return {
    ...commonApi,
    getCoilsForMaterial,
    addCoilToMaterial,
    coilArray,
    getAllCoilFarmCoils,
    allCoilFarmCoils,
    remainingCoilWeightPercentage,
    initialCoilWeightPercentage,
  }
})

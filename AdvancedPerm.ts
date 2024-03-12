// import { computed, ref } from 'vue'
// import { useCheckPermissionsStore } from '@/store/checkPermissions/checkPermissions'

// export function useAdvancedViewPerm() {
//   const checkPermissionsStore = useCheckPermissionsStore()
//   const justARef = ref(false)

//   const hasAdvancedViewPermission = computed(() => {
//     return checkPermissionsStore.hasPermission('hmi.sideMenu.advanced.read')
//       .value
//   })

//   return {
//     checkPermissionsStore,
//     justARef,
//     hasAdvancedViewPermission,
//   }
// }
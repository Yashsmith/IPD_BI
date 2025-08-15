#!/usr/bin/env python3
"""
Fix the sector mapping inconsistency causing baseline methods to return 0.0000 performance
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

def fix_sector_mapping_inconsistency():
    """
    Fix the inconsistent sector mapping between _extract_relevant_sectors 
    and baseline evaluation methods
    """
    
    print("=== FIXING SECTOR MAPPING INCONSISTENCY ===")
    
    # Define the canonical startup-to-sector mapping
    canonical_mapping = {
        'startup_1': 'technology', 'startup_2': 'finance', 'startup_3': 'healthcare',
        'startup_4': 'technology', 'startup_5': 'finance', 'startup_6': 'healthcare', 
        'startup_7': 'retail', 'startup_8': 'technology', 'startup_9': 'finance',
        'startup_10': 'healthcare', 'startup_11': 'technology', 'startup_12': 'finance',
        'startup_13': 'healthcare', 'startup_14': 'retail', 'startup_15': 'technology',
        'startup_16': 'finance', 'startup_17': 'healthcare', 'startup_18': 'retail',
        'startup_19': 'technology', 'startup_20': 'finance', 'startup_21': 'healthcare',
        'startup_22': 'retail', 'startup_23': 'technology', 'startup_24': 'finance',
        'startup_25': 'healthcare', 'startup_26': 'retail', 'startup_27': 'technology',
        'startup_28': 'finance', 'startup_29': 'healthcare', 'startup_30': 'retail',
        'startup_31': 'technology', 'startup_32': 'finance', 'startup_33': 'healthcare',
        'startup_34': 'retail', 'startup_35': 'technology', 'startup_36': 'finance',
        'startup_37': 'healthcare', 'startup_38': 'retail', 'startup_39': 'technology',
        'startup_40': 'finance', 'startup_41': 'healthcare', 'startup_42': 'retail',
        'startup_43': 'technology', 'startup_44': 'finance', 'startup_45': 'healthcare',
        'startup_46': 'retail', 'startup_47': 'technology', 'startup_48': 'finance',
        'startup_49': 'healthcare', 'startup_50': 'retail', 'startup_51': 'technology',
        'startup_52': 'finance', 'startup_53': 'healthcare', 'startup_54': 'retail',
        'startup_55': 'technology', 'startup_56': 'finance', 'startup_57': 'healthcare',
        'startup_58': 'retail', 'startup_59': 'technology', 'startup_60': 'finance',
        'startup_61': 'healthcare', 'startup_62': 'retail', 'startup_63': 'technology',
        'startup_64': 'finance', 'startup_65': 'healthcare', 'startup_66': 'retail',
        'startup_67': 'technology', 'startup_68': 'finance', 'startup_69': 'healthcare',
        'startup_70': 'retail', 'startup_71': 'technology', 'startup_72': 'finance',
        'startup_73': 'healthcare', 'startup_74': 'retail', 'startup_75': 'technology',
        'startup_76': 'finance', 'startup_77': 'healthcare', 'startup_78': 'retail',
        'startup_79': 'technology', 'startup_80': 'finance', 'startup_81': 'healthcare',
        'startup_82': 'retail', 'startup_83': 'technology', 'startup_84': 'finance',
        'startup_85': 'healthcare', 'startup_86': 'retail', 'startup_87': 'technology',
        'startup_88': 'finance', 'startup_89': 'healthcare', 'startup_90': 'retail',
        'startup_91': 'technology', 'startup_92': 'finance', 'startup_93': 'healthcare',
        'startup_94': 'retail', 'startup_95': 'technology', 'startup_96': 'finance',
        'startup_97': 'healthcare', 'startup_98': 'retail', 'startup_99': 'technology',
        'startup_100': 'finance'
    }
    
    # Create a shared sector mapping utility
    sector_mapping_code = f'''
# Shared startup-to-sector mapping for consistent evaluation
STARTUP_TO_SECTOR_MAPPING = {canonical_mapping}

def get_sector_for_startup(startup_id: str) -> str:
    """
    Get the sector for a given startup ID using consistent mapping
    
    Args:
        startup_id: The startup identifier (e.g., 'startup_1')
        
    Returns:
        The sector name (technology, finance, healthcare, retail)
    """
    return STARTUP_TO_SECTOR_MAPPING.get(startup_id, 'technology')
'''
    
    # Write the shared mapping file
    with open('/Users/Sameer/Yashsmith/Research/IPD/bi_platform/src/evaluation/sector_mapping.py', 'w') as f:
        f.write(sector_mapping_code)
    
    print("✅ Created shared sector mapping file: src/evaluation/sector_mapping.py")
    
    # Now we need to update experimental_framework.py to use this consistent mapping
    # Let's read the current file first
    with open('/Users/Sameer/Yashsmith/Research/IPD/bi_platform/src/evaluation/experimental_framework.py', 'r') as f:
        content = f.read()
    
    # Replace the inconsistent sector mapping logic in run_baseline_experiment
    old_mapping_logic = '''                # Convert item IDs to sectors using our mapping
                item_sectors = []
                for item_id in item_ids:
                    if isinstance(item_id, str) and item_id.startswith('startup_'):
                        try:
                            startup_num = int(item_id.split('_')[1])
                            if startup_num <= 25:
                                item_sectors.append('technology')
                            elif startup_num <= 50:
                                item_sectors.append('finance')
                            elif startup_num <= 75:
                                item_sectors.append('healthcare')
                            else:
                                item_sectors.append('retail')
                        except (ValueError, IndexError):
                            # If parsing fails, default to technology
                            item_sectors.append('technology')
                    else:
                        # For non-startup items, map based on item index
                        try:
                            if isinstance(item_id, str) and item_id.startswith('item_'):
                                item_num = int(item_id.split('_')[1])
                            elif isinstance(item_id, int):
                                item_num = item_id
                            else:
                                item_num = 0
                                
                            if item_num <= 25:
                                item_sectors.append('technology')
                            elif item_num <= 50:
                                item_sectors.append('finance')
                            elif item_num <= 75:
                                item_sectors.append('healthcare')
                            else:
                                item_sectors.append('retail')
                        except (ValueError, IndexError):
                            item_sectors.append('technology')'''
    
    new_mapping_logic = '''                # Convert item IDs to sectors using consistent mapping
                from .sector_mapping import get_sector_for_startup
                item_sectors = []
                for item_id in item_ids:
                    item_sectors.append(get_sector_for_startup(item_id))'''
    
    if old_mapping_logic in content:
        content = content.replace(old_mapping_logic, new_mapping_logic)
        print("✅ Updated baseline evaluation sector mapping in experimental_framework.py")
    else:
        print("⚠️  Could not find exact mapping logic to replace - will need manual update")
    
    # Add the import at the top of the file
    if 'from .sector_mapping import get_sector_for_startup' not in content:
        # Find the imports section and add our import
        import_line = "from .sector_mapping import get_sector_for_startup"
        lines = content.split('\n')
        
        # Find the last import line
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_idx = i
        
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_line)
            content = '\n'.join(lines)
            print("✅ Added sector_mapping import to experimental_framework.py")
    
    # Write the updated file
    with open('/Users/Sameer/Yashsmith/Research/IPD/bi_platform/src/evaluation/experimental_framework.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied sector mapping consistency fix to experimental_framework.py")
    
    return True

if __name__ == "__main__":
    fix_sector_mapping_inconsistency()
